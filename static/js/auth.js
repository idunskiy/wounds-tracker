function logout() {
    localStorage.removeItem('token');  // Remove the authentication token
    window.location.href = '/login';  // Redirect to login page
}

function checkAuth() {
    const token = localStorage.getItem('token');
    if (!token) {
        window.location.href = '/login';  // Redirect if not logged in
    }
}

// Automatically check authentication on page load
window.onload = checkAuth;