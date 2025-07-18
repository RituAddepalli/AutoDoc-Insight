<?php
ob_clean(); // ✅ Ensures no unwanted output before JSON
header("Content-Type: application/json"); // ✅ Must come before echo

// Enable error display
ini_set('display_errors', 1);
error_reporting(E_ALL);

// Read input
$data = json_decode(file_get_contents("php://input"), true);
$question = $data["question"] ?? "";

// Upload directory
$uploadDir = __DIR__ . "/uploads/";

// Find latest PDF
$files = array_diff(scandir($uploadDir, SCANDIR_SORT_DESCENDING), ['.', '..']);
$pdfFile = '';
foreach ($files as $file) {
    if (strtolower(pathinfo($file, PATHINFO_EXTENSION)) === 'pdf') {
        $pdfFile = $file;
        break;
    }
}

if ($pdfFile && $question) {
    $pdfPath = realpath($uploadDir . '/' . $pdfFile);
    $safePdf = escapeshellarg($pdfPath);
    $safeQuestion = escapeshellarg($question);

    $pythonExe = __DIR__ . "/venv/Scripts/python.exe";
    $command = "\"$pythonExe\" process_pdf.py $safePdf $safeQuestion";

    // Debug log
    $logFile = __DIR__ . "/log.txt";
    file_put_contents($logFile, "CMD: $command\n", FILE_APPEND);

    $output = shell_exec($command);

    file_put_contents($logFile, "Output:\n$output\n\n", FILE_APPEND);

    if ($output === null || trim($output) === "") {
        $output = "⚠ No output from Python script.";
    }

    $cleanedOutput = trim($output);
    $json = json_encode(["answer" => $cleanedOutput]);

    echo $json;
    exit; // ✅ Prevents further accidental output
} else {
    echo json_encode(["answer" => "❌ No PDF found or question missing."]);
    exit;
}

















