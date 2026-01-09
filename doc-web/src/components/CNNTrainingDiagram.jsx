export default function CNNTrainingDiagram() {
  return (
    <div style={{
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      borderRadius: '12px',
      padding: '40px',
      margin: '30px 0'
    }}>
      <svg
        viewBox="0 0 800 650"
        xmlns="http://www.w3.org/2000/svg"
        style={{
          width: '100%',
          maxWidth: '800px',
          height: 'auto',
          margin: '0 auto',
          display: 'block'
        }}
      >
        {/* Definitions */}
        <defs>
          <linearGradient id="cnnGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#4F46E5', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#7C3AED', stopOpacity: 1 }} />
          </linearGradient>
          <linearGradient id="cnnGrad2" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#10B981', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#059669', stopOpacity: 1 }} />
          </linearGradient>
          <linearGradient id="cnnGrad3" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#F59E0B', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#D97706', stopOpacity: 1 }} />
          </linearGradient>
          <filter id="cnnShadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="4" stdDeviation="8" floodOpacity="0.3"/>
          </filter>
          <marker id="cnnArrowGreen" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#10B981" />
          </marker>
          <marker id="cnnArrowOrange" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#F59E0B" />
          </marker>
          <marker id="cnnArrowPurple" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#A855F7" />
          </marker>
        </defs>

        {/* Title */}
        <text x="400" y="35" fontFamily="Arial, sans-serif" fontSize="22" fontWeight="bold" fill="white" textAnchor="middle">
          Federated CNN Training Architecture
        </text>

        {/* Center Server (Task Management) */}
        <g filter="url(#cnnShadow)">
          <rect x="250" y="70" width="300" height="110" rx="12" fill="url(#cnnGrad1)" />
          <text x="400" y="105" fontFamily="Arial, sans-serif" fontSize="20" fontWeight="bold" fill="white" textAnchor="middle">Center Server</text>
          <text x="400" y="130" fontFamily="Arial, sans-serif" fontSize="14" fill="white" textAnchor="middle" opacity="0.9">(Task Management)</text>
          <text x="400" y="150" fontFamily="Arial, sans-serif" fontSize="12" fill="white" textAnchor="middle" opacity="0.85">Django + RabbitMQ + MinIO</text>
          <text x="400" y="167" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.75">Model Registry &amp; Task Distribution</text>
        </g>

        {/* Arrow: Central to Regional */}
        <line x1="400" y1="180" x2="400" y2="250" stroke="#10B981" strokeWidth="3" markerEnd="url(#cnnArrowGreen)" />
        <text x="425" y="220" fontFamily="Arial, sans-serif" fontSize="13" fill="white" fontWeight="600">RabbitMQ</text>

        {/* Regional Node (Fedevo Server) */}
        <g filter="url(#cnnShadow)">
          <rect x="225" y="270" width="350" height="120" rx="12" fill="url(#cnnGrad2)" />
          <text x="400" y="310" fontFamily="Arial, sans-serif" fontSize="20" fontWeight="bold" fill="white" textAnchor="middle">Regional Node</text>
          <text x="400" y="335" fontFamily="Arial, sans-serif" fontSize="14" fill="white" textAnchor="middle" opacity="0.9">(Fedevo Server)</text>
          <text x="400" y="355" fontFamily="Arial, sans-serif" fontSize="12" fill="white" textAnchor="middle" opacity="0.85">Fed-Evo Strategy + Model Aggregation</text>
          <text x="400" y="375" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.75">Python + Flower + MQTT</text>
        </g>

        {/* Aggregation Return Arrow */}
        <path d="M 580 330 Q 650 330, 650 125 Q 650 80, 550 80"
              stroke="#A855F7"
              strokeWidth="2.5"
              fill="none"
              markerEnd="url(#cnnArrowPurple)"
              strokeDasharray="5,5" />
        <text x="665" y="210" fontFamily="Arial, sans-serif" fontSize="12" fill="white" fontWeight="600">Aggregation</text>

        {/* Arrow: Regional to Devices */}
        <line x1="310" y1="390" x2="170" y2="465" stroke="#F59E0B" strokeWidth="3" markerEnd="url(#cnnArrowOrange)" />
        <line x1="490" y1="390" x2="630" y2="465" stroke="#F59E0B" strokeWidth="3" markerEnd="url(#cnnArrowOrange)" />
        <text x="235" y="430" fontFamily="Arial, sans-serif" fontSize="13" fill="white" fontWeight="600">MQTT</text>
        <text x="560" y="430" fontFamily="Arial, sans-serif" fontSize="13" fill="white" fontWeight="600">MQTT</text>

        {/* Edge Device 1 (Flower Client) */}
        <g filter="url(#cnnShadow)">
          <rect x="50" y="485" width="220" height="140" rx="12" fill="url(#cnnGrad3)" />
          <text x="160" y="520" fontFamily="Arial, sans-serif" fontSize="18" fontWeight="bold" fill="white" textAnchor="middle">Device 1</text>
          <text x="160" y="545" fontFamily="Arial, sans-serif" fontSize="13" fill="white" textAnchor="middle" opacity="0.9">(Flower Client)</text>
          <text x="160" y="570" fontFamily="Arial, sans-serif" fontSize="12" fill="white" textAnchor="middle" opacity="0.85">Local Training</text>
          <text x="160" y="590" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.8">SimpleCNN + MNIST</text>
          <text x="160" y="610" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.75">PyTorch Local Data</text>
        </g>

        {/* Edge Device 2 (Flower Client) */}
        <g filter="url(#cnnShadow)">
          <rect x="530" y="485" width="220" height="140" rx="12" fill="url(#cnnGrad3)" />
          <text x="640" y="520" fontFamily="Arial, sans-serif" fontSize="18" fontWeight="bold" fill="white" textAnchor="middle">Device 2</text>
          <text x="640" y="545" fontFamily="Arial, sans-serif" fontSize="13" fill="white" textAnchor="middle" opacity="0.9">(Flower Client)</text>
          <text x="640" y="570" fontFamily="Arial, sans-serif" fontSize="12" fill="white" textAnchor="middle" opacity="0.85">Local Training</text>
          <text x="640" y="590" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.8">SimpleCNN + MNIST</text>
          <text x="640" y="610" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.75">PyTorch Local Data</text>
        </g>

        {/* Legend */}
        <g transform="translate(50, 640)">
          <text x="0" y="0" fontFamily="Arial, sans-serif" fontSize="12" fill="white" fontWeight="600">Data Flow:</text>
          <line x1="80" y1="-5" x2="120" y2="-5" stroke="#10B981" strokeWidth="2" />
          <text x="130" y="0" fontFamily="Arial, sans-serif" fontSize="11" fill="white">Task Distribution</text>
          <line x1="230" y1="-5" x2="270" y2="-5" stroke="#F59E0B" strokeWidth="2" />
          <text x="280" y="0" fontFamily="Arial, sans-serif" fontSize="11" fill="white">Device Communication</text>
          <line x1="410" y1="-5" x2="450" y2="-5" stroke="#A855F7" strokeWidth="2" strokeDasharray="5,5" />
          <text x="460" y="0" fontFamily="Arial, sans-serif" fontSize="11" fill="white">Model Aggregation</text>
        </g>
      </svg>
    </div>
  )
}
