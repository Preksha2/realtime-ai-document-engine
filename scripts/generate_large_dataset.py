"""
Synthetic event log generator for stress-testing the RAG pipeline.
Generates 50K+ realistic event log entries across multiple services.

Usage:
    python scripts/generate_large_dataset.py
    python scripts/generate_large_dataset.py --num-entries 100000 --output data/large_events
"""
import os
import sys
import random
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SERVICES = [
    "PaymentGateway", "AuthService", "UserService", "OrderService",
    "InventoryService", "NotificationService", "SearchService",
    "AnalyticsEngine", "CacheManager", "LoadBalancer", "APIGateway",
    "SessionManager", "RateLimiter", "DeployManager", "MLPipeline",
    "DataIngestion", "Reconciler", "ReportGenerator", "SecurityModule",
    "MonitoringAgent", "AutoScaler", "Scheduler", "AuditLog",
]

LOG_LEVELS = ["INFO", "WARN", "ERROR", "DEBUG"]
LEVEL_WEIGHTS = [0.70, 0.15, 0.08, 0.07]

USERS = [f"U-{random.randint(1000, 9999)}" for _ in range(500)]
IPS = [f"{random.randint(10,192)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}" for _ in range(200)]
TXNS = [f"TXN-{random.randint(10000, 99999)}" for _ in range(2000)]
JOBS = [f"JOB-{random.randint(1000, 9999)}" for _ in range(300)]
ENDPOINTS = ["/api/v2/transactions", "/api/v2/users", "/api/v2/orders", "/api/v2/search", "/api/v2/reports", "/api/v2/inventory", "/health", "/metrics"]
MODELS = ["fraud_detector_v3", "recommendation_v2", "anomaly_detector_v1", "churn_predictor_v2", "pricing_model_v4"]
DEPLOY_VERSIONS = [f"v{major}.{minor}.{patch}" for major in range(1,4) for minor in range(10,20) for patch in range(0,5)]

TEMPLATES = {
    "request": [
        "Request received: {method} {endpoint} | user_id={user} | session=S-{session}",
        "Request completed: {method} {endpoint} | status={status} | latency={latency}ms",
        "Request failed: {method} {endpoint} | status={error_status} | error={error_msg}",
    ],
    "payment": [
        "Payment validation started | amount={amount} | currency=USD | method={pay_method}",
        "Transaction {txn} approved | auth_code=A-{auth} | latency={latency}ms",
        "Transaction {txn} failed after {retries} retries | error={error_msg} | total_elapsed={elapsed}ms",
        "Fallback triggered: routing to secondary processor | txn={txn}",
        "Refund processed | txn={txn} | amount={amount} | reason={refund_reason}",
        "Payment confirmed | txn={txn} | user_id={user} | total_processing_time={elapsed}ms",
    ],
    "auth": [
        "Login attempt | user_id={user} | ip={ip} | method={auth_method}",
        "Login successful | user_id={user} | session=S-{session} | mfa={mfa}",
        "Failed login attempt | user_id={user} | ip={ip} | reason={fail_reason} | attempt={attempt}/5",
        "Account locked | user_id={user} | ip={ip} | reason=max_attempts_exceeded | lockout=30min",
        "Token refresh | user_id={user} | session=S-{session} | token_age={token_age}min",
        "API key authentication | service={service} | key_id=AK-{key_id} | scope={scope}",
    ],
    "deploy": [
        "Deployment initiated: {service} {version} | env=production | strategy=rolling",
        "Rolling update: pod {pod}/{total_pods} updated | readiness_check={check}",
        "Deployment complete: {service} {version} | total_time={elapsed}s | rollback=not_needed",
        "Post-deployment metrics: error_rate={error_rate}% | p95_latency={latency}ms | p99_latency={p99}ms",
        "Scale-up triggered: adding {replicas} replicas to {service} | reason={scale_reason}",
        "Scale-down executed: {service} reduced to {replicas} replicas | cpu_avg={cpu}%",
    ],
    "ml": [
        "Training job started | model={model} | job_id={job}",
        "Data loaded | source=s3://ml-data/{model} | records={records} | size={size}GB",
        "Training complete | best_epoch={epoch} | train_auc={train_auc} | val_auc={val_auc}",
        "Evaluation on holdout set | test_auc={test_auc} | precision@95recall={precision} | f1={f1}",
        "Model registered: {model} | artifact=s3://models/{model}.tar.gz | status=STAGING",
        "Batch prediction complete | model={model} | flagged={flagged}({flag_pct}%) | avg_latency={latency}ms/record",
    ],
    "infra": [
        "Health check | all services healthy | uptime={uptime}% | active_connections={connections}",
        "CPU spike detected on {node} | cpu={cpu}% | duration={duration}s | threshold=90%",
        "Memory warning on {node} | memory={memory}% | available={available}MB",
        "Disk usage alert | node={node} | disk={disk}% | path=/data",
        "Network latency spike | source={node} | target={target_node} | latency={latency}ms | baseline=2ms",
        "Cache hit ratio | service={service} | ratio={hit_ratio}% | evictions={evictions}/min",
    ],
    "data": [
        "Batch job started: {job_name} | job_id={job}",
        "Loading records from primary DB | date_range={date} | table={table}",
        "Loaded {records} records | size={size}MB | elapsed={elapsed}s",
        "Reconciliation pass complete | matched={matched} | discrepancies={discrepancies}",
        "Report generated: /reports/{report_name} | pages={pages}",
        "Email sent to {recipient} | subject=\"{subject}\"",
    ],
    "security": [
        "Suspicious activity flagged | ip={ip} | reason={sec_reason} | action={sec_action}",
        "Rate limit approaching for IP {ip} | current={current}/{limit} req/min",
        "Rate limit exceeded for IP {ip} | blocked=true | count={current}/{limit} req/min",
        "Brute force pattern detected | user_id={user} | ip={ip} | action=alert_security_team",
        "Security scan complete | vulnerabilities_found={vulns} | critical={critical} | high={high}",
        "Certificate renewal | domain={domain} | expires_in={expires_days} days | auto_renewed={renewed}",
    ],
}


def generate_value(key):
    """Generate a random value for a template placeholder."""
    generators = {
        "method": lambda: random.choice(["GET", "POST", "PUT", "DELETE", "PATCH"]),
        "endpoint": lambda: random.choice(ENDPOINTS),
        "user": lambda: random.choice(USERS),
        "session": lambda: random.randint(10000, 99999),
        "status": lambda: random.choice([200, 201, 204]),
        "error_status": lambda: random.choice([400, 401, 403, 404, 500, 502, 503]),
        "latency": lambda: random.randint(5, 3000),
        "error_msg": lambda: random.choice(["GATEWAY_TIMEOUT", "CONNECTION_REFUSED", "RATE_LIMITED", "INVALID_REQUEST", "INTERNAL_ERROR", "DB_TIMEOUT"]),
        "amount": lambda: round(random.uniform(1.0, 5000.0), 2),
        "pay_method": lambda: random.choice(["credit_card", "debit_card", "paypal", "bank_transfer", "crypto"]),
        "txn": lambda: random.choice(TXNS),
        "auth": lambda: random.randint(10000, 99999),
        "retries": lambda: random.randint(1, 5),
        "elapsed": lambda: random.randint(100, 30000),
        "refund_reason": lambda: random.choice(["customer_request", "duplicate_charge", "fraud_detected", "item_not_received"]),
        "ip": lambda: random.choice(IPS),
        "auth_method": lambda: random.choice(["oauth2", "api_key", "jwt", "saml"]),
        "mfa": lambda: random.choice(["true", "false"]),
        "fail_reason": lambda: random.choice(["invalid_password", "expired_token", "invalid_mfa", "account_suspended"]),
        "attempt": lambda: random.randint(1, 5),
        "token_age": lambda: random.randint(5, 120),
        "service": lambda: random.choice(SERVICES),
        "key_id": lambda: f"{random.randint(1, 200):04d}",
        "scope": lambda: random.choice(["read-only", "read-write", "admin"]),
        "version": lambda: random.choice(DEPLOY_VERSIONS),
        "pod": lambda: random.randint(1, 8),
        "total_pods": lambda: random.choice([4, 6, 8]),
        "check": lambda: random.choice(["passed", "passed (delayed 10s)", "passed (delayed 25s)", "failed_retry_1"]),
        "error_rate": lambda: round(random.uniform(0.0, 2.0), 2),
        "p99": lambda: random.randint(200, 1500),
        "replicas": lambda: random.randint(2, 12),
        "scale_reason": lambda: random.choice(["cpu_threshold", "memory_threshold", "request_queue_depth", "scheduled_scale"]),
        "cpu": lambda: random.randint(20, 95),
        "model": lambda: random.choice(MODELS),
        "job": lambda: random.choice(JOBS),
        "records": lambda: f"{random.randint(10, 5000)},{random.randint(0, 999):03d}",
        "size": lambda: round(random.uniform(0.1, 50.0), 1),
        "epoch": lambda: random.randint(50, 300),
        "train_auc": lambda: round(random.uniform(0.92, 0.99), 4),
        "val_auc": lambda: round(random.uniform(0.90, 0.98), 4),
        "test_auc": lambda: round(random.uniform(0.89, 0.97), 4),
        "precision": lambda: round(random.uniform(0.70, 0.95), 2),
        "f1": lambda: round(random.uniform(0.75, 0.95), 2),
        "flagged": lambda: random.randint(10, 1000),
        "flag_pct": lambda: round(random.uniform(0.01, 5.0), 2),
        "uptime": lambda: round(random.uniform(99.5, 99.99), 2),
        "connections": lambda: random.randint(50, 2000),
        "node": lambda: f"node-{random.randint(1, 20)}",
        "target_node": lambda: f"node-{random.randint(1, 20)}",
        "duration": lambda: random.randint(5, 300),
        "memory": lambda: random.randint(60, 98),
        "available": lambda: random.randint(100, 8000),
        "disk": lambda: random.randint(70, 98),
        "hit_ratio": lambda: round(random.uniform(85, 99.9), 1),
        "evictions": lambda: random.randint(0, 500),
        "job_name": lambda: random.choice(["reconciliation_pipeline", "etl_daily", "report_generation", "data_cleanup", "index_rebuild"]),
        "date": lambda: f"2025-01-{random.randint(1, 31):02d}",
        "table": lambda: random.choice(["transactions", "users", "orders", "sessions", "audit_log"]),
        "matched": lambda: f"{random.randint(10, 100)},{random.randint(0, 999):03d}",
        "discrepancies": lambda: random.randint(0, 50),
        "report_name": lambda: random.choice(["reconciliation", "daily_summary", "fraud_report", "performance_metrics"]),
        "pages": lambda: random.randint(3, 30),
        "recipient": lambda: random.choice(["finance-team@company.com", "ops@company.com", "security-ops@company.com", "ml-team@company.com"]),
        "subject": lambda: random.choice(["Daily Reconciliation Report", "Alert: System Anomaly", "ML Training Complete", "Deployment Summary"]),
        "sec_reason": lambda: random.choice(["rate_limit_burst", "geo_anomaly", "credential_stuffing", "unusual_access_pattern"]),
        "sec_action": lambda: random.choice(["temporary_block_30min", "alert_sent", "captcha_required", "account_flagged"]),
        "current": lambda: random.randint(500, 2000),
        "limit": lambda: random.choice([1000, 2000, 5000]),
        "vulns": lambda: random.randint(0, 20),
        "critical": lambda: random.randint(0, 3),
        "high": lambda: random.randint(0, 8),
        "domain": lambda: random.choice(["api.company.com", "app.company.com", "auth.company.com"]),
        "expires_days": lambda: random.randint(1, 90),
        "renewed": lambda: random.choice(["true", "false"]),
    }
    return generators.get(key, lambda: "unknown")()


def generate_entry(timestamp):
    """Generate a single log entry."""
    level = random.choices(LOG_LEVELS, weights=LEVEL_WEIGHTS, k=1)[0]
    category = random.choice(list(TEMPLATES.keys()))
    template = random.choice(TEMPLATES[category])
    service = random.choice(SERVICES)

    # Fill in template placeholders
    import re
    placeholders = re.findall(r'\{(\w+)\}', template)
    values = {p: generate_value(p) for p in placeholders}
    message = template.format(**values)

    ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return f"[{ts_str}] {level:<5} | {service} | {message}"


def generate_dataset(num_entries: int, output_dir: str, entries_per_file: int = 5000):
    """Generate a large synthetic event log dataset."""
    os.makedirs(output_dir, exist_ok=True)

    start_time = datetime(2025, 1, 1, 0, 0, 0)
    file_count = 0
    total_written = 0

    print(f"Generating {num_entries:,} event log entries...")

    current_batch = []
    for i in range(num_entries):
        # Advance time by 0.5-5 seconds per entry
        start_time += timedelta(seconds=random.uniform(0.5, 5.0))
        entry = generate_entry(start_time)
        current_batch.append(entry)

        if len(current_batch) >= entries_per_file:
            file_count += 1
            date_str = start_time.strftime("%Y%m%d")
            filename = f"events_{date_str}_batch{file_count:03d}.txt"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(current_batch))

            total_written += len(current_batch)
            print(f"  Written: {filepath} ({len(current_batch):,} entries) | Total: {total_written:,}/{num_entries:,}")
            current_batch = []

    # Write remaining entries
    if current_batch:
        file_count += 1
        date_str = start_time.strftime("%Y%m%d")
        filename = f"events_{date_str}_batch{file_count:03d}.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(current_batch))

        total_written += len(current_batch)
        print(f"  Written: {filepath} ({len(current_batch):,} entries) | Total: {total_written:,}/{num_entries:,}")

    print(f"\nDataset generation complete:")
    print(f"  Total entries: {total_written:,}")
    print(f"  Total files: {file_count}")
    print(f"  Output directory: {output_dir}")
    print(f"\nTo index this dataset, run:")
    print(f"  python scripts/demo.py  (update data_dir to '{output_dir}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic event log dataset")
    parser.add_argument("--num-entries", type=int, default=55000, help="Number of log entries to generate (default: 55000)")
    parser.add_argument("--output", type=str, default="data/large_events", help="Output directory")
    parser.add_argument("--entries-per-file", type=int, default=5000, help="Entries per file")
    args = parser.parse_args()
    generate_dataset(args.num_entries, args.output, args.entries_per_file)
