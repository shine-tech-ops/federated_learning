export default function ArchitectureDiagram() {
  return (
    <div style={{
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      borderRadius: '12px',
      padding: '40px',
      margin: '30px 0'
    }}>
      <svg
        viewBox="0 0 800 600"
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
          <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#4F46E5', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#7C3AED', stopOpacity: 1 }} />
          </linearGradient>
          <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#10B981', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#059669', stopOpacity: 1 }} />
          </linearGradient>
          <linearGradient id="grad3" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#F59E0B', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#D97706', stopOpacity: 1 }} />
          </linearGradient>
          <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="4" stdDeviation="8" floodOpacity="0.3"/>
          </filter>
        </defs>

        {/* Central Server */}
        <g filter="url(#shadow)">
          <rect x="250" y="30" width="300" height="120" rx="12" fill="url(#grad1)" />
          <text x="400" y="70" fontFamily="Arial, sans-serif" fontSize="20" fontWeight="bold" fill="white" textAnchor="middle">Central Server</text>
          <text x="400" y="95" fontFamily="Arial, sans-serif" fontSize="14" fill="white" textAnchor="middle" opacity="0.9">Django + PostgreSQL + Redis</text>
          <text x="400" y="115" fontFamily="Arial, sans-serif" fontSize="14" fill="white" textAnchor="middle" opacity="0.9">RabbitMQ + MinIO</text>
          <text x="400" y="135" fontFamily="Arial, sans-serif" fontSize="12" fill="white" textAnchor="middle" opacity="0.8">Task Management</text>
        </g>

        {/* Arrow: Central to Regional */}
        <defs>
          <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#10B981" />
          </marker>
        </defs>
        <line x1="400" y1="150" x2="400" y2="220" stroke="#10B981" strokeWidth="3" markerEnd="url(#arrowhead)" />
        <text x="420" y="190" fontFamily="Arial, sans-serif" fontSize="13" fill="white" fontWeight="600">RabbitMQ</text>

        {/* Regional Node */}
        <g filter="url(#shadow)">
          <rect x="250" y="240" width="300" height="120" rx="12" fill="url(#grad2)" />
          <text x="400" y="280" fontFamily="Arial, sans-serif" fontSize="20" fontWeight="bold" fill="white" textAnchor="middle">Regional Node</text>
          <text x="400" y="305" fontFamily="Arial, sans-serif" fontSize="14" fill="white" textAnchor="middle" opacity="0.9">Python + Fedevo</text>
          <text x="400" y="325" fontFamily="Arial, sans-serif" fontSize="14" fill="white" textAnchor="middle" opacity="0.9">RabbitMQ + MQTT</text>
          <text x="400" y="345" fontFamily="Arial, sans-serif" fontSize="12" fill="white" textAnchor="middle" opacity="0.8">Model Aggregation</text>
        </g>

        {/* Arrow: Regional to Devices */}
        <defs>
          <marker id="arrowhead2" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#F59E0B" />
          </marker>
        </defs>
        <line x1="350" y1="360" x2="200" y2="430" stroke="#F59E0B" strokeWidth="3" markerEnd="url(#arrowhead2)" />
        <line x1="450" y1="360" x2="600" y2="430" stroke="#F59E0B" strokeWidth="3" markerEnd="url(#arrowhead2)" />
        <text x="280" y="400" fontFamily="Arial, sans-serif" fontSize="13" fill="white" fontWeight="600">MQTT</text>
        <text x="510" y="400" fontFamily="Arial, sans-serif" fontSize="13" fill="white" fontWeight="600">MQTT</text>

        {/* Edge Device 1 */}
        <g filter="url(#shadow)">
          <rect x="50" y="450" width="200" height="100" rx="12" fill="url(#grad3)" />
          <text x="150" y="485" fontFamily="Arial, sans-serif" fontSize="18" fontWeight="bold" fill="white" textAnchor="middle">Edge Device 1</text>
          <text x="150" y="510" fontFamily="Arial, sans-serif" fontSize="13" fill="white" textAnchor="middle" opacity="0.9">PyTorch + Fedevo</text>
          <text x="150" y="530" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.8">Local Training</text>
        </g>

        {/* Edge Device 2 */}
        <g filter="url(#shadow)">
          <rect x="300" y="450" width="200" height="100" rx="12" fill="url(#grad3)" />
          <text x="400" y="485" fontFamily="Arial, sans-serif" fontSize="18" fontWeight="bold" fill="white" textAnchor="middle">Edge Device 2</text>
          <text x="400" y="510" fontFamily="Arial, sans-serif" fontSize="13" fill="white" textAnchor="middle" opacity="0.9">PyTorch + Fedevo</text>
          <text x="400" y="530" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.8">Local Training</text>
        </g>

        {/* Edge Device 3 */}
        <g filter="url(#shadow)">
          <rect x="550" y="450" width="200" height="100" rx="12" fill="url(#grad3)" />
          <text x="650" y="485" fontFamily="Arial, sans-serif" fontSize="18" fontWeight="bold" fill="white" textAnchor="middle">Edge Device 3</text>
          <text x="650" y="510" fontFamily="Arial, sans-serif" fontSize="13" fill="white" textAnchor="middle" opacity="0.9">PyTorch + Fedevo</text>
          <text x="650" y="530" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.8">Local Training</text>
        </g>

        {/* Legend */}
        <g transform="translate(50, 570)">
          <text x="0" y="0" fontFamily="Arial, sans-serif" fontSize="12" fill="white" fontWeight="600">Data Flow:</text>
          <line x1="80" y1="-5" x2="120" y2="-5" stroke="#10B981" strokeWidth="2" />
          <text x="130" y="0" fontFamily="Arial, sans-serif" fontSize="11" fill="white">RabbitMQ</text>
          <line x1="200" y1="-5" x2="240" y2="-5" stroke="#F59E0B" strokeWidth="2" />
          <text x="250" y="0" fontFamily="Arial, sans-serif" fontSize="11" fill="white">MQTT</text>
        </g>
      </svg>
    </div>
  )
}
