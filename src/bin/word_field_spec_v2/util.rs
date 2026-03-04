use serde_json::{Map, Value};

pub fn value_to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

pub fn value_to_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_u64().map(|x| x as i64)),
        Value::String(s) => s.parse::<i64>().ok(),
        _ => None,
    }
}

pub fn param_f64(params: &Map<String, Value>, key: &str, default: f64) -> f64 {
    params.get(key).and_then(value_to_f64).unwrap_or(default)
}

pub fn param_i64(params: &Map<String, Value>, key: &str, default: i64) -> i64 {
    params.get(key).and_then(value_to_i64).unwrap_or(default)
}
