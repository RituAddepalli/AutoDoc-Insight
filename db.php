<?php
$host = "localhost";  // XAMPP runs MySQL on localhost
$user = "root";       // Default XAMPP MySQL user
$pass = "";           // No password by default
$dbname = "pdf_ai_db"; // Database name

// Create connection
$conn = new mysqli($host, $user, $pass, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}
?>
