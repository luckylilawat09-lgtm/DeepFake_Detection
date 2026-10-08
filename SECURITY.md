# Security Policy

## Supported Versions

The following table indicates which versions of the DeepFake Detection project receive security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

The DeepFake Detection team takes security vulnerabilities seriously. If you discover a vulnerability or potential security risk in this repository, please report it privately:

1. **Do not disclose the issue publicly** in an open GitHub issue.
2. Open a private security advisory through the repository's GitHub Security tab (preferred), or contact the maintainer directly via email: `luckylilawat09@gmail.com`.
3. Provide a clear description of the vulnerability, including:
   - Steps to reproduce the issue
   - Proof of Concept (PoC) code or requests if available
   - The potential impact and attack vectors

### Response Timeline
- **Initial Response:** Within 48 hours of receipt.
- **Triage & Remediation Plan:** Within 7 business days.
- **Public Disclosure:** Coordinated following a verified patch release.

## Security Best Practices
- Keep your Python environment and dependencies up-to-date.
- Ensure uploaded files are validated and isolated from executable paths.
- Avoid deploying inference endpoints to public networks without authentication or rate limiting.
