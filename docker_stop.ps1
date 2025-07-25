# Docker Stop Script for Autonomous Vehicle Simulation

Write-Host ""
Write-Host "🛑 STOPPING AUTONOMOUS VEHICLE SIMULATION DOCKER SERVICES" -ForegroundColor Red
Write-Host "=" * 60 -ForegroundColor Red
Write-Host ""

Write-Host "Stopping Docker services..." -ForegroundColor Yellow
docker-compose -f docker-compose-simple.yml down --remove-orphans

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ All services stopped successfully!" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "🧹 Cleanup options:" -ForegroundColor Cyan
    Write-Host "Remove all containers: docker-compose -f docker-compose-simple.yml down --rmi all" -ForegroundColor White
    Write-Host "Remove volumes:       docker-compose -f docker-compose-simple.yml down -v" -ForegroundColor White
    Write-Host "Full cleanup:         docker-compose -f docker-compose-simple.yml down -v --rmi all --remove-orphans" -ForegroundColor White
    
}
else {
    Write-Host ""
    Write-Host "❌ Error stopping services" -ForegroundColor Red
    Write-Host "You may need to stop containers manually:" -ForegroundColor Yellow
    Write-Host "docker stop vehicle-simulation data-processor dashboard redis" -ForegroundColor White
}

Write-Host ""
