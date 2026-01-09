export default function DataFlowDiagram() {
  return (
    <div style={{
      background: 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)',
      borderRadius: '12px',
      padding: '40px',
      margin: '30px 0'
    }}>
      <svg
        viewBox="0 0 1200 850"
        xmlns="http://www.w3.org/2000/svg"
        style={{
          width: '100%',
          maxWidth: '1200px',
          height: 'auto',
          margin: '0 auto',
          display: 'block'
        }}
      >
        {/* Definitions */}
        <defs>
          {/* Gradients */}
          <linearGradient id="gradServer" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#667eea', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#764ba2', stopOpacity: 1 }} />
          </linearGradient>
          <linearGradient id="gradRegional" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#06b6d4', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#0891b2', stopOpacity: 1 }} />
          </linearGradient>
          <linearGradient id="gradDevice" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#f59e0b', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#d97706', stopOpacity: 1 }} />
          </linearGradient>
          <linearGradient id="gradStorage" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#10b981', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#059669', stopOpacity: 1 }} />
          </linearGradient>
          <linearGradient id="gradQueue" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#ef4444', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#dc2626', stopOpacity: 1 }} />
          </linearGradient>

          {/* Filters */}
          <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="4" stdDeviation="8" floodOpacity="0.3"/>
          </filter>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>

          {/* Arrow markers */}
          <marker id="arrowBlue" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#3b82f6" />
          </marker>
          <marker id="arrowGreen" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#10b981" />
          </marker>
          <marker id="arrowOrange" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#f59e0b" />
          </marker>
          <marker id="arrowRed" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#ef4444" />
          </marker>
          <marker id="arrowPurple" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#a855f7" />
          </marker>
          <marker id="arrowCyan" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#06b6d4" />
          </marker>
        </defs>

        {/* Title */}
        <text x="600" y="30" fontFamily="Arial, sans-serif" fontSize="24" fontWeight="bold" fill="white" textAnchor="middle">
          Federated Learning Data Flow Architecture
        </text>

        {/* ========== LAYER 1: Central Server Components ========== */}

        {/* PostgreSQL Database */}
        <g filter="url(#shadow)">
          <rect x="50" y="70" width="160" height="90" rx="8" fill="url(#gradStorage)" />
          <text x="130" y="100" fontFamily="Arial, sans-serif" fontSize="16" fontWeight="bold" fill="white" textAnchor="middle">PostgreSQL</text>
          <text x="130" y="120" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">Task Metadata</text>
          <text x="130" y="135" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">Model Versions</text>
          <text x="130" y="150" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">Training Logs</text>
        </g>

        {/* Django Backend */}
        <g filter="url(#shadow)">
          <rect x="250" y="70" width="200" height="90" rx="8" fill="url(#gradServer)" />
          <text x="350" y="100" fontFamily="Arial, sans-serif" fontSize="16" fontWeight="bold" fill="white" textAnchor="middle">Django Backend</text>
          <text x="350" y="120" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">REST API</text>
          <text x="350" y="135" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">Task Management</text>
          <text x="350" y="150" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">Model Registry</text>
        </g>

        {/* MinIO Storage */}
        <g filter="url(#shadow)">
          <rect x="490" y="70" width="160" height="90" rx="8" fill="url(#gradStorage)" />
          <text x="570" y="100" fontFamily="Arial, sans-serif" fontSize="16" fontWeight="bold" fill="white" textAnchor="middle">MinIO</text>
          <text x="570" y="120" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">Model Files</text>
          <text x="570" y="135" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">.pt, .pth, .zip</text>
          <text x="570" y="150" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">S3 Compatible</text>
        </g>

        {/* Redis Cache */}
        <g filter="url(#shadow)">
          <rect x="690" y="70" width="160" height="90" rx="8" fill="url(#gradQueue)" />
          <text x="770" y="100" fontFamily="Arial, sans-serif" fontSize="16" fontWeight="bold" fill="white" textAnchor="middle">Redis</text>
          <text x="770" y="120" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">Session Cache</text>
          <text x="770" y="135" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">Task Status</text>
          <text x="770" y="150" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">Real-time Sync</text>
        </g>

        {/* RabbitMQ */}
        <g filter="url(#shadow)">
          <rect x="890" y="70" width="160" height="90" rx="8" fill="url(#gradQueue)" />
          <text x="970" y="100" fontFamily="Arial, sans-serif" fontSize="16" fontWeight="bold" fill="white" textAnchor="middle">RabbitMQ</text>
          <text x="970" y="120" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">Message Queue</text>
          <text x="970" y="135" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">Task Distribution</text>
          <text x="970" y="150" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">AMQP Protocol</text>
        </g>

        {/* ========== LAYER 2: Regional Node ========== */}

        {/* Main Regional Node Box */}
        <g filter="url(#shadow)">
          <rect x="300" y="270" width="600" height="140" rx="12" fill="url(#gradRegional)" />
          <text x="600" y="305" fontFamily="Arial, sans-serif" fontSize="20" fontWeight="bold" fill="white" textAnchor="middle">Regional Node Server</text>

          {/* Sub-components inside Regional Node */}
          <g>
            <rect x="320" y="320" width="160" height="70" rx="6" fill="rgba(255,255,255,0.15)" />
            <text x="400" y="340" fontFamily="Arial, sans-serif" fontSize="13" fontWeight="bold" fill="white" textAnchor="middle">RabbitMQ Client</text>
            <text x="400" y="358" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.9">Task Subscriber</text>
            <text x="400" y="373" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.9">Status Reporter</text>
          </g>

          <g>
            <rect x="520" y="320" width="160" height="70" rx="6" fill="rgba(255,255,255,0.15)" />
            <text x="600" y="340" fontFamily="Arial, sans-serif" fontSize="13" fontWeight="bold" fill="white" textAnchor="middle">Fedevo Server</text>
            <text x="600" y="358" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.9">Model Aggregator</text>
            <text x="600" y="373" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.9">gRPC Service</text>
          </g>

          <g>
            <rect x="720" y="320" width="160" height="70" rx="6" fill="rgba(255,255,255,0.15)" />
            <text x="800" y="340" fontFamily="Arial, sans-serif" fontSize="13" fontWeight="bold" fill="white" textAnchor="middle">MQTT Broker</text>
            <text x="800" y="358" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.9">Device Manager</text>
            <text x="800" y="373" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.9">Pub/Sub Handler</text>
          </g>
        </g>

        {/* ========== LAYER 3: MQTT Communication ========== */}

        {/* MQTT Topics */}
        <g filter="url(#shadow)">
          <rect x="920" y="480" width="250" height="110" rx="8" fill="rgba(168, 85, 247, 0.2)" stroke="#a855f7" strokeWidth="2" />
          <text x="1045" y="505" fontFamily="Arial, sans-serif" fontSize="14" fontWeight="bold" fill="white" textAnchor="middle">MQTT Topics</text>
          <text x="1045" y="528" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">federated_task_&#123;id&#125;/task_start</text>
          <text x="1045" y="545" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">federated_task_&#123;id&#125;/status</text>
          <text x="1045" y="562" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">federated_task_&#123;id&#125;/model_info</text>
          <text x="1045" y="579" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">federated_task_&#123;id&#125;/complete</text>
        </g>

        {/* ========== LAYER 4: Edge Devices ========== */}

        {/* Edge Device 1 */}
        <g filter="url(#shadow)">
          <rect x="50" y="650" width="220" height="180" rx="10" fill="url(#gradDevice)" />
          <text x="160" y="680" fontFamily="Arial, sans-serif" fontSize="18" fontWeight="bold" fill="white" textAnchor="middle">Edge Device 1</text>

          <rect x="70" y="695" width="180" height="45" rx="4" fill="rgba(255,255,255,0.2)" />
          <text x="160" y="712" fontFamily="Arial, sans-serif" fontSize="12" fill="white" textAnchor="middle">MQTT Client</text>
          <text x="160" y="728" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.9">Task Listener</text>

          <rect x="70" y="750" width="180" height="60" rx="4" fill="rgba(255,255,255,0.2)" />
          <text x="160" y="768" fontFamily="Arial, sans-serif" fontSize="12" fill="white" textAnchor="middle">Fedevo Client</text>
          <text x="160" y="784" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.9">Local Training</text>
          <text x="160" y="799" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.9">PyTorch + MNIST</text>
        </g>

        {/* Edge Device 2 */}
        <g filter="url(#shadow)">
          <rect x="320" y="650" width="220" height="180" rx="10" fill="url(#gradDevice)" />
          <text x="430" y="680" fontFamily="Arial, sans-serif" fontSize="18" fontWeight="bold" fill="white" textAnchor="middle">Edge Device 2</text>

          <rect x="340" y="695" width="180" height="45" rx="4" fill="rgba(255,255,255,0.2)" />
          <text x="430" y="712" fontFamily="Arial, sans-serif" fontSize="12" fill="white" textAnchor="middle">MQTT Client</text>
          <text x="430" y="728" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.9">Task Listener</text>

          <rect x="340" y="750" width="180" height="60" rx="4" fill="rgba(255,255,255,0.2)" />
          <text x="430" y="768" fontFamily="Arial, sans-serif" fontSize="12" fill="white" textAnchor="middle">Fedevo Client</text>
          <text x="430" y="784" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.9">Local Training</text>
          <text x="430" y="799" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.9">PyTorch + CIFAR10</text>
        </g>

        {/* Edge Device 3 */}
        <g filter="url(#shadow)">
          <rect x="590" y="650" width="220" height="180" rx="10" fill="url(#gradDevice)" />
          <text x="700" y="680" fontFamily="Arial, sans-serif" fontSize="18" fontWeight="bold" fill="white" textAnchor="middle">Edge Device 3</text>

          <rect x="610" y="695" width="180" height="45" rx="4" fill="rgba(255,255,255,0.2)" />
          <text x="700" y="712" fontFamily="Arial, sans-serif" fontSize="12" fill="white" textAnchor="middle">MQTT Client</text>
          <text x="700" y="728" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.9">Task Listener</text>

          <rect x="610" y="750" width="180" height="60" rx="4" fill="rgba(255,255,255,0.2)" />
          <text x="700" y="768" fontFamily="Arial, sans-serif" fontSize="12" fill="white" textAnchor="middle">Fedevo Client</text>
          <text x="700" y="784" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.9">Local Training</text>
          <text x="700" y="799" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.9">TensorFlow Lite</text>
        </g>

        {/* ========== Process Flow Indicators ========== */}
        <g>
          <rect x="860" y="650" width="320" height="180" rx="8" fill="rgba(102, 126, 234, 0.15)" stroke="#667eea" strokeWidth="2"/>
          <text x="1020" y="675" fontFamily="Arial, sans-serif" fontSize="15" fill="white" fontWeight="bold" textAnchor="middle">Training Flow</text>

          <text x="880" y="700" fontFamily="Arial, sans-serif" fontSize="11" fill="#10b981" fontWeight="bold">1. Task Creation</text>
          <text x="900" y="715" fontFamily="Arial, sans-serif" fontSize="10" fill="white" opacity="0.9">Backend → RabbitMQ → Regional</text>

          <text x="880" y="738" fontFamily="Arial, sans-serif" fontSize="11" fill="#a855f7" fontWeight="bold">2. Device Notification</text>
          <text x="900" y="753" fontFamily="Arial, sans-serif" fontSize="10" fill="white" opacity="0.9">Regional → MQTT → Devices</text>

          <text x="880" y="776" fontFamily="Arial, sans-serif" fontSize="11" fill="#06b6d4" fontWeight="bold">3. Model Training</text>
          <text x="900" y="791" fontFamily="Arial, sans-serif" fontSize="10" fill="white" opacity="0.9">Devices ↔ Fedevo (gRPC)</text>

          <text x="880" y="814" fontFamily="Arial, sans-serif" fontSize="11" fill="#ef4444" fontWeight="bold">4. Aggregation & Upload</text>
          <text x="900" y="829" fontFamily="Arial, sans-serif" fontSize="10" fill="white" opacity="0.9">Regional → MinIO → Backend</text>
        </g>

      </svg>
    </div>
  )
}
