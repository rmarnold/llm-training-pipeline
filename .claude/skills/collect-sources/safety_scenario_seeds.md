# Safety Scenario Seeds

Language-agnostic seed scenarios for `sota_preference` data generation. Each seed is a starting point — the skill expands these into 20-30 language-specific concrete scenarios per category.

---

## harmful_code_request

Scenarios where the user asks the model to produce code with clearly malicious intent. The model should **refuse** and explain why.

### Seeds

1. **Memory exploitation**: User asks to write code that exploits a buffer overflow or use-after-free vulnerability to gain arbitrary code execution.

2. **Credential harvesting**: User requests a tool that scrapes or brute-forces passwords, API keys, or tokens from systems they don't own.

3. **DoS tooling**: User asks for code that performs denial-of-service attacks — SYN floods, amplification attacks, resource exhaustion against specific targets.

4. **Cryptographic backdoor**: User wants to implement a cryptographic algorithm with an intentional weakness or backdoor for later exploitation.

5. **Privilege escalation**: User requests code to escalate privileges on a system beyond their authorized access level.

6. **Malware components**: User asks for code implementing keyloggers, screen capture without consent, or data exfiltration to a remote server.

7. **Supply chain attack**: User requests modifications to open-source packages that introduce hidden vulnerabilities or backdoors.

8. **Network intrusion**: User asks for code to scan networks, exploit known CVEs, or pivot through systems they don't own.

9. **Data theft**: User requests tools to extract personal data from databases, APIs, or file systems without authorization.

10. **Anti-forensics**: User asks for code that erases logs, timestamps, or other evidence of unauthorized access.

---

## unsafe_code_patterns

Scenarios where the user's code contains security vulnerabilities or unsafe patterns. The user may not have malicious intent, but the code is dangerous. The model should **clarify** the risk and suggest safe alternatives.

### Seeds

1. **Unsafe blocks without justification**: User writes unsafe code (raw pointers, unchecked casts, FFI) without documenting why it's necessary or what invariants must hold.

2. **SQL injection**: User constructs database queries by concatenating user input directly into query strings instead of using parameterized queries.

3. **Deserializing untrusted input**: User deserializes data from network or user input without validation, schema checking, or size limits.

4. **Race conditions**: User has shared mutable state accessed from multiple threads/goroutines without synchronization primitives.

5. **Path traversal**: User constructs file paths from user input without canonicalization or sandboxing, allowing `../../etc/passwd` style attacks.

6. **Hardcoded secrets**: User embeds API keys, passwords, or cryptographic keys directly in source code.

7. **Improper error exposure**: User returns raw error messages (stack traces, internal paths, SQL errors) to end users.

8. **Insecure randomness**: User uses non-cryptographic random number generators for security-sensitive operations (tokens, passwords, nonces).

9. **Missing input validation**: User passes external input directly to system commands, file operations, or network calls without sanitization.

10. **Insecure defaults**: User disables TLS verification, uses HTTP instead of HTTPS, or sets overly permissive CORS policies "for testing" in production-bound code.

---

## uncertainty_calibration

Scenarios where the correct response involves expressing uncertainty, asking for clarification, or acknowledging the limits of the model's knowledge. The model should **hedge** or **clarify** rather than confabulating.

### Seeds

1. **Ambiguous specification**: User provides a feature description that could be implemented multiple valid ways, with different trade-offs. Model should present options, not pick one silently.

2. **Underspecified performance requirements**: User asks to "make it fast" without defining latency targets, throughput requirements, or acceptable trade-offs (memory vs speed).

3. **Conflicting type constraints**: User's type annotations or interface definitions contain contradictions. Model should identify the conflict rather than silently resolving it.

4. **Missing error handling context**: User asks how to handle an error but doesn't specify the error recovery strategy (retry? propagate? log and continue?).

5. **Deprecated API uncertainty**: User asks about an API that may have been deprecated or changed after the model's training cutoff. Model should flag the uncertainty.

6. **Platform-specific behavior**: User assumes behavior that varies across platforms (OS, runtime version, architecture). Model should note the platform dependency.

7. **Incomplete context**: User shares a code snippet without surrounding context (imports, type definitions, module structure). Model should ask for context rather than guessing.

8. **Novel library version**: User references a library version or API that may not have existed during training. Model should express uncertainty about specific version behavior.

9. **Trade-off decisions**: User faces architectural choices (monolith vs microservice, SQL vs NoSQL, sync vs async) where the right answer depends on unstated constraints. Model should ask.

10. **Ambiguous naming**: User references a function, type, or package name that exists in multiple contexts or namespaces. Model should ask which one they mean.
