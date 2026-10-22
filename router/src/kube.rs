//! Just enough of the Kubernetes API: one Lease and one ConfigMap for the dictionary training, and where each pod runs
//! (node, GPU model) for /balancer-status.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde_json::{json, Value};

pub struct Kube {
    http: reqwest::Client,
    base: String,
    pub ns: String,
    /// Read again per call: a projected service account token rotates.
    token_file: Option<String>,
    pub id: String,
    /// client-go's rule: a Lease is expired once its record has not changed for lease_s by *our* clock, so no two
    /// clocks are compared.
    observed: Mutex<Option<(Value, Instant)>>,
}

const SA: &str = "/var/run/secrets/kubernetes.io/serviceaccount";

impl Kube {
    /// `api` overrides the in-cluster address (KUBERNETES_SERVICE_HOST, the service account token and CA); with
    /// neither there is no client.
    pub fn from_env(api: Option<&str>, namespace: Option<&str>) -> Option<Self> {
        let in_cluster = std::env::var("KUBERNETES_SERVICE_HOST")
            .ok()
            .zip(std::env::var("KUBERNETES_SERVICE_PORT").ok());
        let (base, token_file, ca) = match (api, in_cluster) {
            (Some(api), _) => (api.trim_end_matches('/').to_string(), None, None),
            (None, Some((h, p))) => (
                format!("https://{h}:{p}"),
                Some(format!("{SA}/token")),
                std::fs::read(format!("{SA}/ca.crt")).ok(),
            ),
            (None, None) => return None,
        };
        let mut b = reqwest::Client::builder().timeout(Duration::from_secs(5));
        if let Some(c) = ca
            .as_deref()
            .and_then(|c| reqwest::Certificate::from_pem(c).ok())
        {
            b = b.add_root_certificate(c);
        }
        let ns = namespace
            .map(str::to_string)
            .or_else(|| std::fs::read_to_string(format!("{SA}/namespace")).ok())
            .unwrap_or_else(|| "default".into());
        let id = std::env::var("HOSTNAME")
            .unwrap_or_else(|_| format!("zev-router-{:08x}", rand::random::<u32>()));
        Some(Kube {
            http: b.build().expect("kube client"),
            base,
            ns: ns.trim().into(),
            token_file,
            id,
            observed: Mutex::new(None),
        })
    }

    fn req(&self, m: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let r = self.http.request(m, format!("{}{path}", self.base));
        match self
            .token_file
            .as_ref()
            .and_then(|f| std::fs::read_to_string(f).ok())
        {
            Some(t) => r.bearer_auth(t.trim()),
            None => r,
        }
    }

    /// Hold the Lease: create it, renew it, or take it over once expired. Any error = not leader.
    pub async fn lead(&self, lease: &str, lease_s: u64) -> bool {
        use reqwest::{Method, StatusCode};
        let dir = format!("/apis/coordination.k8s.io/v1/namespaces/{}/leases", self.ns);
        let now = chrono::Utc::now()
            .format("%Y-%m-%dT%H:%M:%S%.6fZ")
            .to_string();
        let spec =
            json!({ "holderIdentity": self.id, "leaseDurationSeconds": lease_s, "renewTime": now });
        let got = match self
            .req(Method::GET, &format!("{dir}/{}", lease))
            .send()
            .await
        {
            Ok(r) if r.status() == StatusCode::NOT_FOUND => None,
            Ok(r) if r.status().is_success() => match r.json::<Value>().await {
                Ok(v) => Some(v),
                Err(_) => return false,
            },
            Ok(r) => {
                log::warn!("Lease {}: {}", lease, r.status());
                return false;
            }
            Err(e) => {
                log::warn!("Lease {}: {e}", lease);
                return false;
            }
        };
        let send = match got {
            None => {
                let mut spec = spec;
                spec["acquireTime"] = json!(now);
                self.req(Method::POST, &dir)
                    .json(&json!({ "apiVersion": "coordination.k8s.io/v1", "kind": "Lease", "metadata": { "name": lease }, "spec": spec }))
            }
            Some(mut obj) => {
                let record = obj["spec"].clone();
                let holder = record["holderIdentity"].as_str().unwrap_or("");
                let fresh = {
                    let mut o = self.observed.lock().unwrap();
                    if o.as_ref().is_none_or(|(r, _)| *r != record) {
                        *o = Some((record.clone(), Instant::now()));
                    }
                    o.as_ref().unwrap().1.elapsed() < Duration::from_secs(lease_s)
                };
                if !holder.is_empty() && holder != self.id && fresh {
                    return false;
                }
                if holder != self.id {
                    obj["spec"]["acquireTime"] = json!(now);
                }
                for k in ["holderIdentity", "leaseDurationSeconds", "renewTime"] {
                    obj["spec"][k] = spec[k].clone();
                }
                // metadata.resourceVersion rides along: a replica that renewed in between makes this a 409.
                self.req(Method::PUT, &format!("{dir}/{}", lease))
                    .json(&obj)
            }
        };
        send.send().await.is_ok_and(|r| r.status().is_success())
    }

    pub async fn put_configmap(&self, name: &str, key: &str, dict: &[u8]) -> Result<(), String> {
        use reqwest::{Method, StatusCode};
        let dir = format!("/api/v1/namespaces/{}/configmaps", self.ns);
        let b64 = base64::engine::general_purpose::STANDARD.encode(dict);
        let r = self
            .req(Method::GET, &format!("{dir}/{}", name))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let send = if r.status() == StatusCode::NOT_FOUND {
            self.req(Method::POST, &dir).json(&json!({
                "apiVersion": "v1", "kind": "ConfigMap", "metadata": { "name": name }, "binaryData": { (key): b64 },
            }))
        } else if r.status().is_success() {
            let mut cm: Value = r.json().await.map_err(|e| e.to_string())?;
            if let Some(d) = cm.get_mut("data").and_then(Value::as_object_mut) {
                d.remove(key);
            }
            cm["binaryData"][key] = json!(b64);
            self.req(Method::PUT, &format!("{dir}/{}", name)).json(&cm)
        } else {
            return Err(format!("GET ConfigMap {}: {}", name, r.status()));
        };
        let r = send.send().await.map_err(|e| e.to_string())?;
        if r.status().is_success() {
            Ok(())
        } else {
            Err(format!("ConfigMap {}: {}", name, r.status()))
        }
    }

    /// Pod IP -> (node, GPU model from the node's nvidia.com/gpu.product label), for the pods of this namespace. Needs
    /// `list pods` here and `get nodes`; without them, or for a rented box, nothing is shown.
    pub async fn placement(&self) -> Result<HashMap<IpAddr, (String, Option<String>)>, String> {
        let get = |path: String| async move {
            let r = self
                .req(reqwest::Method::GET, &path)
                .send()
                .await
                .map_err(|e| e.to_string())?;
            if !r.status().is_success() {
                return Err(format!("GET {path}: {}", r.status()));
            }
            r.json::<Value>().await.map_err(|e| e.to_string())
        };
        let pods = get(format!(
            "/api/v1/namespaces/{}/pods?fieldSelector=status.phase%3DRunning",
            self.ns
        ))
        .await?;
        let mut gpus: HashMap<String, Option<String>> = HashMap::new();
        let mut out = HashMap::new();
        for p in pods["items"].as_array().into_iter().flatten() {
            let (Some(ip), Some(node)) = (
                p["status"]["podIP"].as_str().and_then(|i| i.parse().ok()),
                p["spec"]["nodeName"].as_str(),
            ) else {
                continue;
            };
            if !gpus.contains_key(node) {
                let gpu = get(format!("/api/v1/nodes/{node}"))
                    .await
                    .ok()
                    .and_then(|n| {
                        n["metadata"]["labels"]["nvidia.com/gpu.product"]
                            .as_str()
                            .map(str::to_string)
                    });
                gpus.insert(node.to_string(), gpu);
            }
            out.insert(ip, (node.to_string(), gpus[node].clone()));
        }
        Ok(out)
    }
}
