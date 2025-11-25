# Contributing to HELIOS

## Full Agent Development Strategy

HELIOS adopts a **"Full Agent"** development approach, where AI coding agents are the primary contributors to the codebase. This strategy aims to maintain code consistency and reduce errors caused by AI misunderstanding human-written code.

### Core Principles

1. **AI-First Development**: Code is written primarily by AI agents (GitHub Copilot, Claude, etc.) operating under supervision
2. **Human Read-Only**: Human developers review and supervise, but do not directly modify the code
3. **Supervised Autonomy**: AI agents work autonomously within strict guidelines defined in `.github/copilot-instructions.md`
4. **Consistency**: AI-generated code maintains uniform patterns, reducing cognitive overhead for AI agents in future modifications

### Why Full Agent?

**Reduces AI Comprehension Errors**: When AI agents modify human-written code, they may misinterpret intent, naming conventions, or architectural decisions. By keeping the codebase AI-native, we minimize these misunderstandings.

**Enforces Standards**: AI agents strictly follow documented patterns (units, testing, documentation) defined in project instructions, ensuring consistent quality.

**Maintains Context**: AI-written code includes extensive comments and docstrings optimized for AI comprehension in future sessions.

**Audit Trail**: Every modification is logged in `agent-logs/` with detailed descriptions, creating a transparent development history.

## How to Contribute

### As a Human Contributor

1. **Review & Supervise**: Examine AI-generated code for correctness and scientific validity
2. **Provide Requirements**: Write detailed specifications in natural language
3. **Validate Tests**: Ensure unit tests verify physical coherence, not just code execution
4. **Report Issues**: Open GitHub issues describing problems or desired features
5. **Guide Architecture**: Propose architectural decisions that AI agents will implement

**Do NOT**:
- ❌ Directly edit Python source files in `src/helios/`
- ❌ Modify tests without AI agent involvement
- ❌ Bypass the agent logging system

### As an AI Agent

Follow instructions in `.github/copilot-instructions.md`. Every contribution must include:

1. **Docstrings**: Numpy-style documentation for all public functions
2. **Comments**: Clear English explanations of non-obvious logic
3. **Unit Tests**: Validate correctness AND physical coherence
4. **Modification Log**: Create `agent-logs/YYYY.MM.DD-NN_<topic>.md`

#### Modification Log Format

```markdown
# Agent Modification Log: <Brief Title>

**Date**: YYYY-MM-DD  
**Session**: NN  
**Topic**: <topic-keyword>  
**Agent**: <AI Agent Name/Model>

## Summary
Brief description of changes

## Files Created
- List of new files

## Files Modified
- List of modified files

## Changes Made
Detailed description of modifications

## Tests Added/Updated
- Test files and what they validate

## Breaking Changes
Any API changes or migration notes
```

### Workflow

1. **Issue Creation**: Human or AI creates GitHub issue describing task
2. **AI Implementation**: AI agent implements following `.github/copilot-instructions.md`
3. **Self-Testing**: AI agent runs tests and validates physical coherence
4. **Logging**: AI agent creates modification log in `agent-logs/`
5. **Pull Request**: AI agent or human creates PR with changes
6. **Human Review**: Human validates scientific correctness and architectural soundness
7. **Merge**: Upon approval, changes merge to main branch

## Code Quality Standards

### Mandatory for Every Function

- ✅ **English docstring** (numpy-style for Sphinx)
- ✅ **Unit test** in `tests/` or embedded in module
- ✅ **Physical validation** in tests (units, magnitudes, conservation laws)
- ✅ **Comments** explaining non-trivial logic

### Example

```python
def calculate_stellar_flux(distance: u.Quantity, luminosity: u.Quantity) -> u.Quantity:
    """
    Calculate stellar flux at given distance.
    
    Parameters
    ----------
    distance : astropy.Quantity
        Distance to star (parsecs, AU, or meters)
    luminosity : astropy.Quantity
        Stellar luminosity (solar luminosities or watts)
    
    Returns
    -------
    astropy.Quantity
        Flux in W/m²
    
    Notes
    -----
    Uses inverse square law: F = L / (4π d²)
    """
    # Convert to SI units for calculation
    d_m = distance.to(u.m)
    L_W = luminosity.to(u.W)
    
    # Inverse square law
    flux = L_W / (4 * np.pi * d_m**2)
    
    return flux.to(u.W / u.m**2)

def test_calculate_stellar_flux():
    """Test flux calculation with physical validation."""
    # Sun at 1 AU should give solar constant (~1361 W/m²)
    flux = calculate_stellar_flux(1*u.AU, 1*u.L_sun)
    solar_constant = 1361 * u.W / u.m**2
    
    # Allow 1% tolerance for constants
    assert np.abs(flux - solar_constant) / solar_constant < 0.01
    
    # Test inverse square law: 2x distance = 1/4 flux
    flux_1au = calculate_stellar_flux(1*u.AU, 1*u.L_sun)
    flux_2au = calculate_stellar_flux(2*u.AU, 1*u.L_sun)
    assert np.abs(flux_1au / flux_2au - 4.0) < 0.01
```

## Testing Philosophy

Tests must verify:
1. **Code executes**: No runtime errors
2. **Physical coherence**: Results match expected physics (units, magnitudes, scaling laws)
3. **Edge cases**: Boundary conditions, zero values, extreme inputs

## Documentation

- **Auto-generated**: Sphinx builds API docs from docstrings
- **Manual**: Add `.md` files in `docs/` for tutorials and guides
- **Notebooks**: Place examples in `examples/` with explanatory markdown

## Questions?

Open a GitHub issue or discussion. Describe your question in detail, and an AI agent or human maintainer will respond.

## License

HELIOS is released under the MIT License. By contributing, you agree to license your contributions under the same terms.
