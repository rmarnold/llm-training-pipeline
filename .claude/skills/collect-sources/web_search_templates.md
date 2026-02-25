# Web Search Query Templates

Per-language, per-capability search query templates for Phase 1 source collection. Claude adapts these templates into concrete queries — do NOT use verbatim. Always include the language name in the actual query.

---

## Rust

### language_semantics
Covers: ownership, borrowing, lifetimes, type system, trait system, generics, macros.

Templates:
- `Rust ownership and borrowing explained with examples`
- `Rust lifetime annotations guide`
- `Rust trait system and generics tutorial`
- `Rust macro_rules declarative macros patterns`
- `Rust type system advanced features associated types`
- `Rust move semantics vs copy semantics`
- `Rust borrow checker common errors and fixes`
- `Rust PhantomData and advanced lifetime patterns`

### std_library
Covers: Iterator, Vec, HashMap, String, Result, Option, io, fs, collections.

Templates:
- `Rust standard library Iterator trait methods guide`
- `Rust Vec HashMap collections best practices`
- `Rust String str conversion patterns`
- `Rust Result Option error handling combinators`
- `Rust std::io read write patterns`
- `Rust std::fs file operations examples`
- `Rust std::collections BTreeMap HashSet usage`
- `Rust standard library unstable features nightly`

### ecosystem_crates
Covers: serde, tokio, clap, reqwest, tracing, anyhow, thiserror, rayon, axum.

Templates:
- `Rust serde serialize deserialize custom implementation`
- `Rust tokio async runtime tutorial`
- `Rust clap command line argument parsing v4`
- `Rust reqwest HTTP client examples`
- `Rust tracing structured logging guide`
- `Rust anyhow thiserror error handling patterns`
- `Rust rayon parallel iterators guide`
- `Rust axum web framework tutorial`

### idiomatic_patterns
Covers: Result chains, iterator combinators, pattern matching, builder pattern, newtype, From/Into.

Templates:
- `Rust idiomatic error handling Result chain patterns`
- `Rust iterator combinators map filter fold collect`
- `Rust pattern matching exhaustive enum examples`
- `Rust builder pattern implementation`
- `Rust newtype pattern and From Into conversions`
- `Rust functional programming style idioms`
- `Rust avoiding clone performance idiomatic alternatives`
- `Rust API design guidelines and conventions`

---

## Python

### language_semantics
Covers: dynamic typing, decorators, context managers, metaclasses, descriptors, GIL, async/await.

Templates:
- `Python decorators advanced patterns functools.wraps`
- `Python context managers __enter__ __exit__ contextlib`
- `Python metaclasses and class creation`
- `Python descriptor protocol __get__ __set__`
- `Python GIL global interpreter lock explained`
- `Python async await asyncio patterns`
- `Python data model dunder methods guide`
- `Python closures and scoping rules LEGB`

### std_library
Covers: collections, pathlib, typing, itertools, functools, dataclasses, json, re.

Templates:
- `Python collections module Counter defaultdict deque`
- `Python pathlib Path operations guide`
- `Python typing module advanced type hints`
- `Python itertools recipes and patterns`
- `Python functools lru_cache partial reduce`
- `Python dataclasses advanced features`
- `Python json serialization custom encoders`
- `Python re regex patterns and best practices`

### ecosystem_packages
Covers: requests, pandas, fastapi, pydantic, pytest, sqlalchemy, numpy.

Templates:
- `Python requests library advanced usage sessions`
- `Python pandas DataFrame operations best practices`
- `Python FastAPI tutorial REST API endpoints`
- `Python pydantic v2 model validation`
- `Python pytest fixtures parametrize markers`
- `Python SQLAlchemy 2.0 ORM patterns`
- `Python numpy array operations vectorization`
- `Python click typer CLI framework comparison`

### idiomatic_patterns
Covers: comprehensions, generators, EAFP, context managers, unpacking, walrus operator.

Templates:
- `Python list dict set comprehensions advanced`
- `Python generator expressions and yield patterns`
- `Python EAFP vs LBYL coding style`
- `Python unpacking starred expressions`
- `Python walrus operator := use cases`
- `Python Pythonic code best practices PEP 8`
- `Python dataclass vs namedtuple vs TypedDict`
- `Python protocol classes structural subtyping`

### type_checking
Covers: mypy strict mode, Protocol, generics, TypeVar, ParamSpec, overload.

Templates:
- `Python mypy strict mode configuration guide`
- `Python typing Protocol structural subtyping`
- `Python generic types TypeVar bound constraints`
- `Python ParamSpec Callable type hints`
- `Python overload decorator typing patterns`
- `Python type narrowing isinstance TypeGuard`
- `Python mypy common errors and fixes`
- `Python typing best practices for large codebases`

---

## TypeScript

### language_semantics
Covers: type system, generics, conditional types, mapped types, template literals, declaration merging.

Templates:
- `TypeScript type system advanced features`
- `TypeScript generics constraints and defaults`
- `TypeScript conditional types infer keyword`
- `TypeScript mapped types and utility types`
- `TypeScript template literal types patterns`
- `TypeScript declaration merging interfaces`
- `TypeScript type narrowing and type guards`
- `TypeScript satisfies operator use cases`

### std_library
Covers: DOM APIs, Node.js built-ins, fetch, Promises, Web APIs.

Templates:
- `TypeScript DOM manipulation typed APIs`
- `TypeScript Node.js built-in modules fs path`
- `TypeScript fetch API typed responses`
- `TypeScript Promise async patterns advanced`
- `TypeScript Web APIs typed examples`
- `TypeScript node:test built-in test runner`
- `TypeScript streams and buffers Node.js`
- `TypeScript EventEmitter typed events`

### ecosystem_packages
Covers: express, react, zod, prisma, next.js, drizzle, tRPC.

Templates:
- `TypeScript Express middleware typed handlers`
- `TypeScript React hooks typed patterns`
- `TypeScript Zod schema validation tutorial`
- `TypeScript Prisma ORM type-safe queries`
- `TypeScript Next.js App Router patterns`
- `TypeScript drizzle-orm typed queries`
- `TypeScript tRPC end-to-end type safety`
- `TypeScript effect-ts functional programming`

### idiomatic_patterns
Covers: discriminated unions, type guards, branded types, exhaustive switches, assertion functions.

Templates:
- `TypeScript discriminated unions pattern matching`
- `TypeScript user-defined type guards`
- `TypeScript branded types nominal typing`
- `TypeScript exhaustive switch never type`
- `TypeScript assertion functions asserts keyword`
- `TypeScript readonly and immutable patterns`
- `TypeScript builder pattern with generics`
- `TypeScript error handling patterns Result type`

---

## Go

### language_semantics
Covers: goroutines, channels, interfaces, embedding, defer, error handling, generics.

Templates:
- `Go goroutines and channels patterns`
- `Go interfaces implicit implementation`
- `Go struct embedding and composition`
- `Go defer panic recover patterns`
- `Go error handling wrapping unwrapping`
- `Go generics type parameters constraints`
- `Go context package usage patterns`
- `Go memory model and synchronization`

### std_library
Covers: net/http, encoding/json, io, fmt, sync, testing, os, strings.

Templates:
- `Go net/http server handler patterns`
- `Go encoding/json marshal unmarshal custom`
- `Go io Reader Writer interface patterns`
- `Go sync Mutex WaitGroup patterns`
- `Go testing package table-driven tests`
- `Go os file operations exec command`
- `Go strings bytes manipulation`
- `Go text/template html/template guide`

### ecosystem_packages
Covers: gin, cobra, viper, gorm, zap, wire, testify.

Templates:
- `Go gin web framework middleware tutorial`
- `Go cobra CLI application building`
- `Go viper configuration management`
- `Go gorm ORM database patterns`
- `Go zap structured logging guide`
- `Go wire dependency injection`
- `Go testify assertions mock suite`
- `Go sqlx database query patterns`

### idiomatic_patterns
Covers: error wrapping, table-driven tests, functional options, interface segregation.

Templates:
- `Go error wrapping fmt.Errorf %w patterns`
- `Go table-driven tests with subtests`
- `Go functional options pattern`
- `Go interface segregation small interfaces`
- `Go constructor patterns NewFoo conventions`
- `Go package organization best practices`
- `Go concurrency patterns fan-out fan-in`
- `Go graceful shutdown signal handling`

### linting
Covers: golangci-lint rules, go vet, staticcheck, common lint fixes.

Templates:
- `Go golangci-lint configuration and rules`
- `Go vet common issues and fixes`
- `Go staticcheck analysis patterns`
- `Go code review checklist and standards`
- `Go common linting errors and solutions`

---

## Site Priority (per language)

### Rust
1. `doc.rust-lang.org` (The Book, Reference, Rustonomicon, std docs)
2. `docs.rs` (crate documentation)
3. `rust-lang.github.io` (Rust by Example, Clippy lints)
4. `blog.rust-lang.org` (edition guides, RFCs)
5. `without.boats`, `smallcultfollowing.com` (core team blogs)

### Python
1. `docs.python.org` (official docs, PEPs, tutorial)
2. `mypy.readthedocs.io` (type checking)
3. `realpython.com` (tutorials)
4. `peps.python.org` (PEP documents)
5. `fastapi.tiangolo.com`, `docs.pydantic.dev` (ecosystem)

### TypeScript
1. `www.typescriptlang.org` (official docs, handbook)
2. `nodejs.org/docs` (Node.js API docs)
3. `developer.mozilla.org` (MDN Web Docs)
4. `react.dev` (React docs)
5. `nextjs.org/docs` (Next.js docs)

### Go
1. `go.dev` (official docs, blog, tour)
2. `pkg.go.dev` (package documentation)
3. `gobyexample.com` (examples)
4. `go.dev/blog` (official blog)
5. `golangci-lint.run` (linting docs)

---

## Known Permissive Docs Sites

These sites serve documentation under known permissive licenses. Use the listed license when fetching from these domains:

| Domain | License | Notes |
|--------|---------|-------|
| `doc.rust-lang.org` | MIT/Apache-2.0 | All Rust project docs |
| `rust-lang.github.io` | MIT/Apache-2.0 | Rust by Example, etc. |
| `docs.rs` | MIT/Apache-2.0 | Auto-generated crate docs (license of each crate may differ) |
| `docs.python.org` | PSF-2.0 | Python Software Foundation license (permissive) |
| `peps.python.org` | PSF-2.0 | PEP documents |
| `www.typescriptlang.org` | Apache-2.0 | TypeScript docs |
| `developer.mozilla.org` | CC-BY-SA-2.5 | MDN Web Docs (attribution required) |
| `nodejs.org` | MIT | Node.js docs |
| `go.dev` | BSD-3-Clause | Go project docs |
| `pkg.go.dev` | BSD-3-Clause | Go package docs |
| `gobyexample.com` | CC-BY-3.0 | Go by Example (attribution required) |
| `react.dev` | CC-BY-4.0 | React docs (attribution required) |
| `nextjs.org` | MIT | Next.js docs |
| `fastapi.tiangolo.com` | MIT | FastAPI docs |
| `docs.pydantic.dev` | MIT | Pydantic docs |
| `mypy.readthedocs.io` | MIT | mypy docs |
| `realpython.com` | unknown_needs_review | Commercial site, check per article |
| `golangci-lint.run` | GPL-3.0 | NOTE: docs site is GPL — content needs review |
