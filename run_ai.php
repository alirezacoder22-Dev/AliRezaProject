<?php
set_time_limit(0);
$cwd = '/home/whiir/public_html';
$binary = $cwd . '/main_x86';
$outputFile = $cwd . '/ai_output.txt';

$cmd = "cd " . escapeshellarg($cwd) . " && " 
     . escapeshellarg($binary) 
     . " > " . escapeshellarg($outputFile) . " 2>&1";

$pid = pcntl_fork();
if ($pid == 0) {
    pcntl_exec('/bin/bash', ['-c', $cmd]);
    exit(1);
} else {
    pcntl_waitpid($pid, $status);
    sleep(1);
    if (file_exists($outputFile)) {
        echo file_get_contents($outputFile);
    } else {
        echo "Output file not created.\n";
    }
}
