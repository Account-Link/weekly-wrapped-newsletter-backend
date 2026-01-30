# Database Reset & Migrate (Docker, no psql in image)

Steps to wipe the Postgres schema and apply the current single-head Alembic migration.

```bash
# 1) Drop and recreate the public schema
sudo docker compose run --rm web sh -lc "uv run python - <<'PY'
import os, sqlalchemy as sa
engine = sa.create_engine(os.environ['DATABASE_URL'], future=True)
with engine.begin() as conn:
    conn.exec_driver_sql('DROP SCHEMA public CASCADE')
    conn.exec_driver_sql('CREATE SCHEMA public')
    conn.exec_driver_sql('GRANT ALL ON SCHEMA public TO public')
print('schema reset')
PY"

# 2) Apply migrations (single head 0001_initial)
sudo docker compose run --rm web sh -lc "uv run alembic upgrade head"

# 3) Start services
sudo docker compose up --build -d --scale cron-worker=4 --scale cron-scheduler=0
```

Notes:
- This wipes all data.
- Alembic head: `0001_initial`.
```
