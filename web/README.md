# HELIOS Web Interface

This directory contains the source code for the HELIOS Web Interface, a modern React-based application for interacting with the HELIOS simulation framework.

## Prerequisites

- **Docker Desktop** (or Docker Engine + Compose)
- Nothing else! (Python and Node.js are handled inside Docker)

## Getting Started

1. **Start the Application**:
   Open a terminal in the root `HELIOS` directory (parent of this one) and run:
   ```bash
   docker-compose up
   ```

2. **Access the Interface**:
   Open your browser and navigate to:
   - **http://localhost:3000** for the Web Interface
   - **http://localhost:8000/docs** for the API Documentation

## Development

### Backend (`/backend`)
Built with **FastAPI**.
- `app.py`: Main application entry point.
- `/simulate`: Endpoint to run simulations.

### Frontend (`/frontend`)
Built with **React**, **Vite**, and **TailwindCSS**.
- Interface for configuring Stars, Planets, and Telescopes.
- Visualizes the simulation output.

## Troubleshooting

- **Ports occupied**: Ensure ports 3000 and 8000 are free.
- **Docker errors**: Make sure Docker Desktop is running.
