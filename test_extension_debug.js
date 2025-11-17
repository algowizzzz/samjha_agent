// Test URL construction like the extension does
const baseUrl = "http://host.docker.internal:8000/";
const path = "/api/doc_review/documents";

try {
    const url = new URL(path, baseUrl);
    console.log("✓ URL construction works:");
    console.log("  Base:", baseUrl);
    console.log("  Path:", path);
    console.log("  Result:", url.toString());
} catch (error) {
    console.log("✗ URL construction failed:");
    console.log("  Error:", error.message);
}

// Test with params
try {
    const url = new URL("/api/doc_review/vfs/stat", baseUrl);
    url.searchParams.set("file_id", "test123");
    url.searchParams.set("path", "/");
    console.log("\n✓ URL with params works:");
    console.log("  Result:", url.toString());
} catch (error) {
    console.log("\n✗ URL with params failed:");
    console.log("  Error:", error.message);
}


