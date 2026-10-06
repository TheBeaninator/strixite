# Contributing to strixite

Thank you for wanting to help. Pull requests and issues are welcome.

## How a pull request lands

I test every change on my own Strix Halo before it goes in: the full test suite, and a before/after benchmark for
anything that touches speed. If it holds up, I merge it. It can take a few days.

strixite is published from the repository I work in day to day, so after a merge the next release may rearrange your
change a little to fit where the engine has moved since. It stays your change, and your name stays on it.

Contributions are accepted under the project's license, [AGPL-3.0](LICENSE).

Please don't include code, comments or docs copied from other projects. strixite is written from scratch, and I'd
like to keep it that way. Ideas from elsewhere are fine; say where they came from.

## What helps a change get in

- **Measurements from a Strix Halo**, with how you took them: the command, the settings, and both the old and the
  new path run on the same machine, ideally alternated.
- **Output identity:** whether greedy output is unchanged, and if not, where and why it differs.
- **A test** that fails without your change, where that's possible.
- **Small, focused changes.** One idea per pull request is much easier for me to verify.

## Scope

strixite targets one chip and one model: AMD Strix Halo (gfx1151) on Linux, and Qwen3.8-Flash-Next. I can't take on
ports to other platforms or models here, but forks are very welcome (see [Platform](README.md#platform)).
