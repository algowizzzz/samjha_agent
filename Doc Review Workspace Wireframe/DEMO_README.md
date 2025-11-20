# 📄 Doc Review Demo - OSFI CAR Chapter 1

## 🎯 What's This?

A **standalone demo** of the Doc Review application with the OSFI CAR Chapter 1 document pre-loaded. Perfect for showcasing all features without manual setup!

## ✨ Pre-loaded Features

- ✅ **Document**: CAR_Chapter_1_First_5_Pages (71 blocks)
- ✅ **Comments**: 12 saved comments with yellow highlights
- ✅ **AI Suggestions**: 10 AI suggestions with blue highlights  
- ✅ **Chat History**: Persistent conversation history
- ✅ **Document Analysis**: Full markdown analysis in left panel
- ✅ **Dark Mode**: Toggle with moon icon
- ✅ **Floating Toolbar**: Text selection toolbar with formatting
- ✅ **Auto-save**: Changes save every 5 seconds

## 🚀 Quick Start

### Option 1: Automated (Recommended)

```bash
# From this directory
./run-demo.sh
```

Then open: **http://localhost:3001/demo.html**

### Option 2: Manual

```bash
# Terminal 1: Start backend
cd /Users/saadahmed/samjha_agent/samjha_agent
python web/app.py

# Terminal 2: Serve demo
cd "Doc Review Workspace Wireframe/build"
python3 -m http.server 3001
```

Then open: **http://localhost:3001/demo.html**

## 📦 What's Included

### File Structure
```
build/
├── demo.html         # 862KB standalone file (CSS + JS embedded)
├── index.html        # Original build (requires assets/)
└── assets/
    ├── index-*.css   # Styles
    └── index-*.js    # App logic
```

### Pre-configured Settings
- **Auto-loads**: CAR_Chapter_1_First_5_Pages document
- **localStorage**: Document ID and page state saved
- **API URL**: Points to localhost:8000

## 🎮 Demo Features

### Left Panel (Analysis)
- Click "Analyze Document" to regenerate analysis
- Rich markdown formatting with color-coded sections
- Collapsible sections

### Center Panel (Editor)
- Select text → Floating toolbar appears
- Comment on text → Yellow highlight
- Ask AI → Blue suggestion highlight
- Save button shows: Save → Saving... → Saved ✓

### Right Panel (Chat)
- Chat history persists across refreshes
- Ask questions about the document
- Dark mode toggle (affects whole page)
- Export conversation

## 🔧 Troubleshooting

### Backend Not Running?
```bash
cd /Users/saadahmed/samjha_agent/samjha_agent
python web/app.py
```

### Port 3001 Already in Use?
```bash
# Use a different port
python3 -m http.server 3002
# Then open http://localhost:3002/demo.html
```

### Document Not Loading?
1. Check backend is running: `curl http://localhost:8000/health`
2. Check browser console for errors (F12)
3. Clear localStorage and refresh

## 🌟 Key Improvements in This Demo

Compared to the original, this version has:
- ✅ Chat history persistence (NEW!)
- ✅ Precise character-level highlighting (NEW!)
- ✅ Full-page dark mode (NEW!)
- ✅ Enhanced save button states (NEW!)
- ✅ Color picker in toolbar (NEW!)
- ✅ Highlight toggle button (NEW!)
- ✅ Editable prompts page (NEW!)
- ✅ Auto-save every 5 seconds (NEW!)

## 📸 Screenshots

### Light Mode
- Clean, minimal interface
- Yellow comment highlights
- Blue AI suggestion highlights

### Dark Mode  
- Full-page dark theme
- All panels adapt to dark colors
- Better for extended viewing

## 🚢 Deployment

To deploy this demo:

1. **Copy the demo file**:
   ```bash
   cp build/demo.html /your/web/server/
   ```

2. **Ensure backend is accessible**:
   - Update `window.API_BASE_URL` in demo.html if needed
   - Or deploy backend alongside frontend

3. **Serve the file**:
   - Any static file server works
   - Nginx, Apache, S3, Netlify, etc.

## 📝 Notes

- **File Size**: 862KB (includes all CSS and JS)
- **Backend Required**: Yes (for document data, comments, chat)
- **Browser Support**: Modern browsers (Chrome, Firefox, Safari, Edge)
- **No Build Step**: Just serve the HTML file

## 🎓 What to Demo

1. **Show document already loaded** ✨
2. **Select text → Comment** (yellow highlight)
3. **Select text → Ask AI** (blue suggestion)
4. **Click Save** (shows Saving... → Saved ✓)
5. **Open chat → Ask question** (history persists)
6. **Toggle dark mode** (whole page)
7. **Show left panel analysis** (rich formatting)
8. **Refresh page** (everything stays loaded!) 🚀

---

Built with ❤️ using React, Lexical, and Tailwind CSS

