# Contributing to NubAgent

Thank you for your interest in contributing to NubAgent! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Welcome newcomers and help them learn
- Keep discussions professional and on-topic

## Getting Started

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/nubagent.git
   cd nubagent
   ```
3. **Install dependencies**:
   ```bash
   npm install
   ```
4. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your values
   ```
5. **Start development server**:
   ```bash
   npm run dev
   ```

## Development Setup

### Prerequisites

- Node.js 18+ 
- npm 9+
- Git

### Project Structure

```
nubagent/
├── api/                    # Backend API endpoints
│   ├── middleware/         # Express middleware
│   └── tools/              # API tool implementations
├── public/                 # Static assets
├── src/
│   ├── components/         # React components
│   │   ├── Chat/          # Chat-related components
│   │   ├── Search/        # Search components
│   │   ├── Library/       # Library components
│   │   ├── Settings/      # Settings components
│   │   └── UI/            # Reusable UI components
│   ├── hooks/             # Custom React hooks
│   ├── lib/               # Utility functions
│   ├── store/             # Zustand stores
│   └── index.css          # Global styles
├── index.html             # HTML template
└── package.json           # Dependencies
```

## Coding Standards

### JavaScript/React

- Use ES6+ features
- Functional components with hooks
- Proper error handling
- JSDoc comments for complex functions

```jsx
/**
 * Button component with multiple variants
 * @param {object} props - Component props
 * @param {string} props.variant - Button style
 * @param {function} props.onClick - Click handler
 */
export function Button({ variant = 'primary', onClick, children }) {
  return (
    <button className={`btn btn--${variant}`} onClick={onClick}>
      {children}
    </button>
  );
}
```

### Naming Conventions

- **Components**: PascalCase (`ChatMessage`, `SearchResults`)
- **Files**: Match component name or descriptive (PascalCase for components, camelCase for utilities)
- **Variables/Functions**: camelCase (`isLoading`, `handleClick`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_FILE_SIZE`, `API_BASE`)
- **CSS Classes**: BEM-like with double dashes (`chat-message--user`, `btn--primary`)

### Code Organization

1. **Imports** (in order):
   - React and core libraries
   - Third-party libraries
   - Internal imports (absolute paths)
   - Relative imports
   - CSS imports

2. **Component Structure**:
   - PropTypes/type definitions
   - Component function
   - Return statement
   - Default exports

### State Management

- Use Zustand for global state
- Use React Query for server state
- Keep local state in components when appropriate

```javascript
// Global state (Zustand)
import { useChatStore } from '../store/useChatStore';

// Server state (React Query)
import { useQuery } from '@tanstack/react-query';
```

## Pull Request Process

### Before Submitting

1. **Update documentation** if changing functionality
2. **Add tests** for new features
3. **Run linting**: `npm run lint`
4. **Test the build**: `npm run build`
5. **Test locally** with different scenarios

### PR Title Format

```
type(scope): description

Examples:
feat(chat): add message reactions
fix(search): handle empty results
docs(readme): update installation steps
```

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tested locally
- [ ] Added tests
- [ ] Updated existing tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added where needed
- [ ] Documentation updated
```

## Testing

### Running Tests

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run specific test file
npm test -- ChatMessage.test.jsx
```

### Writing Tests

```jsx
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from './Button';

describe('Button', () => {
  it('renders children correctly', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByText('Click me')).toBeInTheDocument();
  });

  it('calls onClick when clicked', () => {
    const handleClick = jest.fn();
    render(<Button onClick={handleClick}>Click</Button>);
    fireEvent.click(screen.getByText('Click'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });
});
```

## Documentation

### Code Comments

- Use JSDoc for functions and complex logic
- Add inline comments for non-obvious code
- Keep comments up-to-date

### README Updates

Update README.md when:
- Adding new features
- Changing installation steps
- Modifying API endpoints
- Adding configuration options

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Questions about contributing

## License

By contributing, you agree that your contributions will be licensed under the project's license.
