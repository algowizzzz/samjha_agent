<!DOCTYPE html>
<!--
Copyright All rights Reserved 2025-2030, Ashutosh Sinha, Email: ajsinha@gmail.com
-->
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error - SAJHA MCP Server</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Bootstrap Icons -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    <!-- Custom CSS -->
    <link href="/static/css/style.css?v=20251113" rel="stylesheet">
    
    
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="/login">
                <i class="bi bi-bank"></i> BMO Risk & Finance Agentic Platform
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav me-auto">
                    
                </ul>
                <ul class="navbar-nav">
                    
                    <li class="nav-item">
                        <a class="nav-link" href="/login">
                            <i class="bi bi-box-arrow-in-right"></i> Login
                        </a>
                    </li>
                    
                </ul>
            </div>
        </div>
    </nav>
    
    <!-- Main Content -->
    <main class="container-fluid py-4">
        
<div class="container">
    <div class="row justify-content-center">
        <div class="col-md-6">
            <div class="card border-danger">
                <div class="card-header bg-danger text-white">
                    <h4 class="mb-0"><i class="bi bi-exclamation-triangle-fill"></i> Page Not Found</h4>
                </div>
                <div class="card-body text-center">
                    <p class="lead">The requested page does not exist</p>
                    
                    <div class="mt-4">
                        <a href="/dashboard" class="btn btn-primary">
                            <i class="bi bi-house"></i> Go to Dashboard
                        </a>
                        <button onclick="window.history.back()" class="btn btn-secondary">
                            <i class="bi bi-arrow-left"></i> Go Back
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

    </main>
    
    <!-- Footer -->
    <footer class="footer mt-auto py-3 bg-light">
        <div class="container text-center">
            <span class="text-muted">© 2025 BMO Risk & Finance Agentic Platform</span>
        </div>
    </footer>
    
    <!-- jQuery -->
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <!-- Socket.IO -->
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <!-- Custom JS -->
    <script src="/static/js/main.js?v=20251113"></script>
    
    
</body>
</html>