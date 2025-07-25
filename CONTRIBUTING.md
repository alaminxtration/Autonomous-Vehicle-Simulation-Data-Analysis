# Contributing to Autonomous Vehicle Simulation

Thank you for your interest in contributing to this project! 🚗

## Getting Started

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Autonomous-Vehicle-Simulation-Data-Analysis.git
   cd Autonomous-Vehicle-Simulation-Data-Analysis
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Start Redis for development**
   ```bash
   docker-compose -f docker-compose-minimal.yml up -d
   ```

## Development Workflow

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run tests
   python -m pytest tests/
   
   # Test specific components
   python test_redis_integration.py
   python deployment_status.py
   
   # Format code
   black .
   
   # Lint code
   flake8 .
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**

## Code Style

### Python Style Guide
- Follow PEP 8
- Use Black for code formatting
- Maximum line length: 88 characters
- Use type hints where appropriate

### Commit Message Convention
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/changes
- `refactor:` for code refactoring

## Testing

### Running Tests
```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_simulation.py

# Integration tests
python test_redis_integration.py
python deployment_status.py
```

### Writing Tests
- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Include integration tests for complex features

## Documentation

### Code Documentation
- Use docstrings for all functions and classes
- Include parameter and return type information
- Provide usage examples for public APIs

### README Updates
- Update README.md for new features
- Include code examples
- Update installation instructions if needed

## Areas for Contribution

### 🚀 High Priority
- Additional vehicle types and sensors
- Advanced ML algorithms for risk assessment
- Performance optimizations
- Extended Docker deployment options

### 🔧 Medium Priority
- Additional dashboard visualizations
- Data export/import features
- Configuration management improvements
- Monitoring and alerting features

### 📚 Low Priority
- Documentation improvements
- Code cleanup and refactoring
- Additional test coverage
- Example notebooks and tutorials

## Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Review**: All pull requests are reviewed by maintainers

## Recognition

Contributors will be recognized in:
- README.md contributors section
- CHANGELOG.md for significant contributions
- GitHub contributor graphs

Thank you for helping make autonomous vehicle simulation better! 🎉
