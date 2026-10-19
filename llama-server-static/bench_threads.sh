#!/bin/bash

CLI="/home/rubikair/public_html/build/bin/llama-server"
MODEL="/home/rubikair/public_html/smollm.gguf"
PORT=8090
LOG="/home/rubikair/public_html/llama_server_bench.log"

echo "=== BENCHMARK STARTED ===" | tee -a $LOG

# لیست تعداد رشته‌هایی که میخوایم تست کنیم
for THREADS in 2 4 8 16 32; do
    echo -e "\n--- Testing with $THREADS threads ---" | tee -a $LOG
    
    # کشتن سرور قبلی
    pkill -f "llama-server.*$PORT" 2>/dev/null
    sleep 2
    
    # اجرای سرور با تعداد رشته جدید
    $CLI -m $MODEL --port $PORT -t $THREADS --host 0.0.0.0 -c 512 > /dev/null 2>&1 &
    
    # صبر برای لود شدن مدل
    sleep 8
    
    # ارسال درخواست تست (تولید ۵۰ توکن)
    RESULT=$(curl -s -X POST "http://127.0.0.1:$PORT/completion" \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Write a short story about a robot:", "n_predict": 50, "temperature": 0.1}')
    
    # استخراج سرعت از خروجی JSON
    SPEED=$(echo $RESULT | grep -o '"predicted_per_second":[0-9.]*' | cut -d: -f2)
    
    if [ -z "$SPEED" ]; then
        echo "❌ Failed to get speed for $THREADS threads" | tee -a $LOG
    else
        echo "✅ Threads: $THREADS | Speed: $SPEED tokens/sec" | tee -a $LOG
    fi
done

# کشتن سرور در پایان
pkill -f "llama-server.*$PORT" 2>/dev/null
echo -e "\n=== BENCHMARK FINISHED ===" | tee -a $LOG
