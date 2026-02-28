# Deploy Backend for WeChat Mini Program

This guide covers Step 1: exposing the backend on a public HTTPS domain.

## 1. Server and DNS

1. Prepare a Linux server with Docker and Docker Compose plugin installed.
2. Open inbound ports `80` and `443`.
3. Point your DNS A record to the server IP, for example:
   - `api.your-domain.com` -> `<your_server_ip>`

## 2. Configure environment

1. Copy app environment:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and set at least:
   - `LLM_PROVIDER`
   - provider API key (`ZHIPU_API_KEY` or `OPENAI_API_KEY`)
3. Copy deploy environment:
   ```bash
   cp .env.deploy.example .env.deploy
   ```
4. Edit `.env.deploy`:
   - `DOMAIN=api.your-domain.com`
   - `ACME_EMAIL=you@example.com`

## 3. Start production stack

```bash
docker compose -f docker-compose.prod.yml --env-file .env.deploy up -d --build
```

## 4. Verify HTTPS and API

```bash
curl https://api.your-domain.com/health
curl https://api.your-domain.com/api/status
```

If both return JSON, Step 1 is done.

## 5. WeChat Mini Program config

In WeChat public platform:

1. Go to `Development` -> `Development Settings`.
2. Add `https://api.your-domain.com` as request legal domain.

Then Mini Program can call:

- `POST /api/ask` (recommended first integration endpoint)

## Troubleshooting

1. Certificate not issued:
   - verify DNS resolves to the server
   - verify ports 80/443 are reachable
2. API unreachable:
   - run `docker compose -f docker-compose.prod.yml ps`
   - run `docker compose -f docker-compose.prod.yml logs -f caddy`
   - run `docker compose -f docker-compose.prod.yml logs -f tabletop-agent`
