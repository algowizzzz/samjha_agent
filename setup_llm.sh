#!/bin/bash
# Quick LLM Setup Script for SAJHA MCP Server

echo "🚀 SAJHA MCP Server - LLM Setup"
echo "================================"
echo ""

# Check if .env exists
if [ -f ".env" ]; then
    echo "✓ .env file exists"
else
    echo "📝 Creating .env from .env.example..."
    cp .env.example .env
    echo "✓ Created .env file"
fi

# Check for API key
if grep -q "sk-your-openai-api-key-here" .env 2>/dev/null; then
    echo ""
    echo "⚠️  Please add your API key to .env file:"
    echo "   OPENAI_API_KEY=sk-your-actual-key-here"
    echo ""
    echo "   Get your key from: https://platform.openai.com/api-keys"
    echo ""
elif grep -q "OPENAI_API_KEY=sk-" .env 2>/dev/null; then
    echo "✓ OpenAI API key found in .env"
elif grep -q "ANTHROPIC_API_KEY=sk-ant-" .env 2>/dev/null; then
    echo "✓ Anthropic API key found in .env"
else
    echo "⚠️  No API key found in .env"
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo ""
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Install dependencies
echo ""
echo "📦 Installing LLM dependencies..."
pip install -q langchain langchain-openai langchain-anthropic langchain-core openai anthropic langgraph

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API key (if not done)"
echo "2. Run: python run_server.py"
echo "3. Open: http://localhost:8000/agent/chat"
echo ""
echo "See LLM_SETUP.md for detailed instructions"
