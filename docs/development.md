# Development Guide

## Project Structure

```
federated_learning/
├── backend/          # Django backend
├── frontend/         # Vue 3 frontend
├── regional/         # Regional node service
├── device/           # Edge device client
├── shared/           # Shared model definitions
└── docs/             # Documentation
```

## Development Setup

### Backend Development

```bash
cd backend
pip install -r requirements.txt

# Setup database
python manage.py migrate

# Run development server
python manage.py runserver
```

### Frontend Development

```bash
cd frontend
pnpm install

# Development server
pnpm dev

# Build for production
pnpm build
```

### Regional Node Development

```bash
cd regional
pip install -r requirements.txt

# Configure environment (see regional/readme.md)
python run.py
```

### Device Development

```bash
cd device
pip install -r requirements.txt

# Configure device/config.py
python start_device.py <device_id> <region_id> <central_server_url>
```

## Core Concepts

### Federated Learning Flow

1. **Task Creation**: Central server creates a federated learning task
2. **Task Distribution**: Task sent to regional node via RabbitMQ
3. **Device Notification**: Regional node notifies devices via MQTT
4. **Training**: Devices connect to fedevo server and train locally
5. **Aggregation**: fedevo server aggregates model updates
6. **Result Upload**: Aggregated model uploaded to central server

### Model Management

Models are stored in MinIO and managed through the backend API:
- Upload models via frontend or API
- Models are versioned
- Models can be downloaded by regional nodes and devices

### Task Types

1. **CNN Tasks**: Standard federated learning, auto-collect models after completion
2. **Proxy Model Tasks**: Different execution flow
3. **Large Model Tasks**: Execute large models with parameter desensitization

## Code Structure

### Backend

- `learn_management/`: Task and model management
- `user/`: User management
- `utils/`: Common utilities and constants

### Regional Node

- `app/fed/`: fedevo server management
- `app/service/`: Task management
- `app/utils/`: Communication clients (RabbitMQ, MQTT, HTTP)

### Device

- `flower_client.py`: fedevo client implementation
- `mqtt_handler.py`: MQTT message handling
- `main.py`: Main device logic

## Adding New Features

### Adding a New Aggregation Algorithm

1. Implement aggregation function in regional node
2. Add algorithm to backend constants
3. Update frontend to show new option

### Adding a New Model Type

1. Define model in `shared/` or device-specific location
2. Update model loading logic
3. Update frontend model selection

### Adding New Communication Protocol

1. Implement client in `app/utils/` (regional) or device
2. Update configuration
3. Update documentation

## Testing

### Backend Tests

```bash
cd backend
python manage.py test
```

### Frontend Tests

```bash
cd frontend
pnpm test:unit
```

### Integration Testing

Use the test script:
```bash
python test_federated_learning.py
```

## Debugging

### Backend Logs

```bash
docker-compose logs -f backend
```

### Regional Node Logs

Check `logs/regional.log` or console output.

### Device Logs

Check device log files configured in `device/config.py`.

### MQTT Debugging

Use MQTT client tools to monitor topics:
- Task topics: `federated_task_{device_id}/task_start`
- Status topics: `federated_task_{device_id}/status`

## Best Practices

1. **Error Handling**: Always handle network errors and timeouts
2. **Logging**: Use structured logging (loguru)
3. **Configuration**: Use environment variables for sensitive data
4. **Code Style**: Follow Python PEP 8 and Vue style guide
5. **Documentation**: Update docs when adding features

## Common Issues

### Device Not Connecting

- Check MQTT broker connectivity
- Verify device ID and region ID
- Check network connectivity to regional node

### Training Not Starting

- Verify fedevo server is running on regional node
- Check device can reach regional node
- Verify model download succeeded

### Model Upload Failing

- Check MinIO connectivity
- Verify credentials
- Check disk space

## API Documentation

Backend provides REST API. Check API endpoints:
- Task management: `/api/tasks/`
- Model management: `/api/models/`
- Node management: `/api/nodes/`

Use frontend to explore API or check backend code for endpoint definitions.
