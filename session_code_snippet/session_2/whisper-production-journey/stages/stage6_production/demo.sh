#!/bin/bash
# ===========================================
# Docker Compose Demo Script
# ===========================================
# 
# This script demonstrates the production deployment
# with nginx load balancer and multiple workers.
#
# Usage:
#   ./demo.sh start      # Start with 1 worker
#   ./demo.sh scale 3    # Scale to 3 workers
#   ./demo.sh test       # Run load test (15 users, 60s)
#   ./demo.sh logs       # Watch worker logs
#   ./demo.sh stop       # Stop everything

set -e
cd "$(dirname "$0")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

case "$1" in
    start)
        print_status "Building and starting services..."
        docker compose -f docker-compose.demo.yml build
        docker compose -f docker-compose.demo.yml up -d
        
        print_status "Waiting for services to be healthy..."
        sleep 5
        
        # Wait for whisper-api to be ready
        for i in {1..60}; do
            if curl -s http://localhost/health > /dev/null 2>&1; then
                print_success "Services are ready!"
                echo ""
                echo "  🌐 API:      http://localhost/docs"
                echo "  📊 Health:   http://localhost/health/full"
                echo ""
                exit 0
            fi
            echo -n "."
            sleep 2
        done
        print_warning "Services may still be starting. Check with: docker compose -f docker-compose.demo.yml logs"
        ;;
        
    scale)
        WORKERS=${2:-3}
        print_status "Scaling to $WORKERS workers..."
        docker compose -f docker-compose.demo.yml up -d --scale whisper-api=$WORKERS
        sleep 3
        
        print_success "Scaled to $WORKERS workers"
        echo ""
        docker compose -f docker-compose.demo.yml ps
        ;;
        
    test)
        USERS=${2:-15}
        DURATION=${3:-60s}
        
        print_status "Running load test: $USERS users for $DURATION"
        echo ""
        
        # Check if locust is available locally
        if command -v locust &> /dev/null; then
            locust -f ../stage2_load_test_fail/locustfile.py \
                   --host=http://localhost \
                   --headless \
                   --users $USERS \
                   --spawn-rate 5 \
                   --run-time $DURATION \
                   --only-summary
        else
            print_warning "Locust not found locally. Starting Locust in Docker..."
            docker compose -f docker-compose.demo.yml --profile testing up locust -d
            echo ""
            echo "  🦗 Locust Web UI: http://localhost:8089"
            echo ""
            echo "  Configure your test in the web UI."
        fi
        ;;
        
    logs)
        print_status "Watching worker logs (Ctrl+C to stop)..."
        echo ""
        docker compose -f docker-compose.demo.yml logs -f whisper-api | grep --color=auto -E "(POST /transcribe|request_id|processing_seconds)"
        ;;
        
    metrics)
        print_status "Fetching metrics from workers..."
        curl -s http://localhost/metrics
        ;;
        
    stop)
        print_status "Stopping all services..."
        docker compose -f docker-compose.demo.yml --profile testing down
        print_success "Services stopped"
        ;;
        
    clean)
        print_warning "Removing all containers and volumes..."
        docker compose -f docker-compose.demo.yml --profile testing down -v
        print_success "Cleaned up"
        ;;
        
    *)
        echo "Usage: $0 {start|scale N|test [users] [duration]|logs|metrics|stop|clean}"
        echo ""
        echo "Commands:"
        echo "  start        Build and start with 1 worker"
        echo "  scale N      Scale to N workers"
        echo "  test         Run load test (default: 15 users, 60s)"
        echo "  logs         Watch worker logs"
        echo "  metrics      Fetch Prometheus metrics"
        echo "  stop         Stop all services"
        echo "  clean        Remove containers and volumes"
        echo ""
        echo "Demo Flow:"
        echo "  1. $0 start"
        echo "  2. $0 test           # Single worker under load"
        echo "  3. $0 scale 3"
        echo "  4. $0 test           # 3 workers under load"
        echo "  5. $0 logs           # Watch distribution"
        exit 1
        ;;
esac
