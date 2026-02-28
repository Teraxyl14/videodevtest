# Font Assets for Caption Rendering

Place the following font files here before building the Docker image:

- **Impact.ttf** — Primary TikTok/MrBeast style font
- **Montserrat-Black.ttf** — Alternative high-impact font (download from Google Fonts)

These fonts are COPIED into the Docker container at build time via Dockerfile.
Without them, captions will fail with a clear error instead of silently falling back to 11px.

## Getting the fonts

1. **Impact.ttf**: Copy from `C:\Windows\Fonts\impact.ttf`
2. **Montserrat-Black.ttf**: Download from https://fonts.google.com/specimen/Montserrat
