export default function CloudEdgeDiagram() {
  return (
    <div style={{
      background: 'linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%)',
      borderRadius: '12px',
      padding: '40px',
      margin: '30px 0'
    }}>
      <svg
        viewBox="0 0 1000 750"
        xmlns="http://www.w3.org/2000/svg"
        style={{
          width: '100%',
          maxWidth: '1000px',
          height: 'auto',
          margin: '0 auto',
          display: 'block'
        }}
      >
        {/* Definitions */}
        <defs>
          <linearGradient id="cloudGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#3B82F6', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#2563EB', stopOpacity: 1 }} />
          </linearGradient>
          <linearGradient id="cloudGrad2" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#10B981', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#059669', stopOpacity: 1 }} />
          </linearGradient>
          <linearGradient id="cloudGrad3" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#8B5CF6', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#7C3AED', stopOpacity: 1 }} />
          </linearGradient>
          <linearGradient id="cloudGrad4" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#F59E0B', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#D97706', stopOpacity: 1 }} />
          </linearGradient>
          <filter id="cloudShadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="4" stdDeviation="8" floodOpacity="0.3"/>
          </filter>
          <marker id="cloudArrowBlue" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#3B82F6" />
          </marker>
          <marker id="cloudArrowGreen" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#10B981" />
          </marker>
          <marker id="cloudArrowOrange" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#F59E0B" />
          </marker>
          <marker id="cloudArrowPurple" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#8B5CF6" />
          </marker>
        </defs>

        {/* Title */}
        <text x="500" y="35" fontFamily="Arial, sans-serif" fontSize="24" fontWeight="bold" fill="white" textAnchor="middle">
          Cloud-Edge Model Collaboration Architecture
        </text>

        {/* ========== Central Server (API) ========== */}
        <g filter="url(#cloudShadow)">
          <rect x="200" y="70" width="600" height="260" rx="12" fill="url(#cloudGrad1)" />
          <text x="500" y="105" fontFamily="Arial, sans-serif" fontSize="22" fontWeight="bold" fill="white" textAnchor="middle">Central Server (API)</text>

          {/* API Endpoints */}
          <g>
            <rect x="230" y="125" width="540" height="100" rx="8" fill="rgba(255,255,255,0.15)" />
            <text x="500" y="150" fontFamily="Arial, sans-serif" fontSize="15" fill="white" textAnchor="middle" fontWeight="bold">RESTful API Endpoints</text>
            <text x="330" y="175" fontFamily="Arial, sans-serif" fontSize="12" fill="white" opacity="0.9">• POST /api/models/upload</text>
            <text x="330" y="195" fontFamily="Arial, sans-serif" fontSize="12" fill="white" opacity="0.9">• GET /api/models/download</text>
            <text x="330" y="215" fontFamily="Arial, sans-serif" fontSize="12" fill="white" opacity="0.9">• POST /api/data/upload</text>
            <text x="580" y="175" fontFamily="Arial, sans-serif" fontSize="12" fill="white" opacity="0.9">• GET /api/models/list</text>
            <text x="580" y="195" fontFamily="Arial, sans-serif" fontSize="12" fill="white" opacity="0.9">• Flask + Python</text>
            <text x="580" y="215" fontFamily="Arial, sans-serif" fontSize="12" fill="white" opacity="0.9">• HTTP/REST Protocol</text>
          </g>

          {/* Storage Components */}
          <g>
            <rect x="250" y="245" width="230" height="70" rx="6" fill="rgba(16, 185, 129, 0.3)" stroke="#10B981" strokeWidth="2" />
            <text x="365" y="270" fontFamily="Arial, sans-serif" fontSize="14" fill="white" fontWeight="bold">Model Storage (MinIO)</text>
            <text x="365" y="290" fontFamily="Arial, sans-serif" fontSize="11" fill="white" opacity="0.9">Large Models</text>
            <text x="365" y="305" fontFamily="Arial, sans-serif" fontSize="11" fill="white" opacity="0.9">Compressed Models</text>
          </g>

          <g>
            <rect x="520" y="245" width="230" height="70" rx="6" fill="rgba(139, 92, 246, 0.3)" stroke="#8B5CF6" strokeWidth="2" />
            <text x="635" y="270" fontFamily="Arial, sans-serif" fontSize="14" fill="white" fontWeight="bold">Data Storage (Database)</text>
            <text x="635" y="290" fontFamily="Arial, sans-serif" fontSize="11" fill="white" opacity="0.9">Input-Output Pairs</text>
            <text x="635" y="305" fontFamily="Arial, sans-serif" fontSize="11" fill="white" opacity="0.9">Training Data</text>
          </g>
        </g>

        {/* ========== HTTP/REST API Connection ========== */}
        <text x="500" y="365" fontFamily="Arial, sans-serif" fontSize="16" fill="white" textAnchor="middle" fontWeight="600">HTTP/REST API</text>

        {/* Arrows: Server to Devices */}
        <line x1="350" y1="330" x2="230" y2="475" stroke="#3B82F6" strokeWidth="3" markerEnd="url(#cloudArrowBlue)" />
        <line x1="650" y1="330" x2="770" y2="475" stroke="#3B82F6" strokeWidth="3" markerEnd="url(#cloudArrowBlue)" />

        <text x="270" y="405" fontFamily="Arial, sans-serif" fontSize="13" fill="#10B981" fontWeight="600">Model Download</text>
        <text x="720" y="405" fontFamily="Arial, sans-serif" fontSize="13" fill="#10B981" fontWeight="600">Model Download</text>

        {/* ========== Edge Devices ========== */}

        {/* Edge Device 1 */}
        <g filter="url(#cloudShadow)">
          <rect x="50" y="495" width="280" height="230" rx="12" fill="url(#cloudGrad4)" />
          <text x="190" y="530" fontFamily="Arial, sans-serif" fontSize="20" fontWeight="bold" fill="white" textAnchor="middle">Device 1</text>
          <text x="190" y="555" fontFamily="Arial, sans-serif" fontSize="14" fill="white" textAnchor="middle" opacity="0.9">(Edge)</text>

          {/* Components inside Device */}
          <rect x="70" y="570" width="240" height="60" rx="6" fill="rgba(255,255,255,0.15)" />
          <text x="190" y="590" fontFamily="Arial, sans-serif" fontSize="13" fill="white" textAnchor="middle" fontWeight="bold">EdgeDeviceClient</text>
          <text x="190" y="608" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">• Download compressed model</text>
          <text x="190" y="623" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">• Run inference locally</text>

          <rect x="70" y="640" width="240" height="70" rx="6" fill="rgba(255,255,255,0.15)" />
          <text x="190" y="660" fontFamily="Arial, sans-serif" fontSize="13" fill="white" textAnchor="middle" fontWeight="bold">Data Collection</text>
          <text x="190" y="678" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">• Collect input-output pairs</text>
          <text x="190" y="693" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">• Upload to central server</text>
        </g>

        {/* Edge Device 2 */}
        <g filter="url(#cloudShadow)">
          <rect x="670" y="495" width="280" height="230" rx="12" fill="url(#cloudGrad4)" />
          <text x="810" y="530" fontFamily="Arial, sans-serif" fontSize="20" fontWeight="bold" fill="white" textAnchor="middle">Device 2</text>
          <text x="810" y="555" fontFamily="Arial, sans-serif" fontSize="14" fill="white" textAnchor="middle" opacity="0.9">(Edge)</text>

          {/* Components inside Device */}
          <rect x="690" y="570" width="240" height="60" rx="6" fill="rgba(255,255,255,0.15)" />
          <text x="810" y="590" fontFamily="Arial, sans-serif" fontSize="13" fill="white" textAnchor="middle" fontWeight="bold">EdgeDeviceClient</text>
          <text x="810" y="608" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">• Download compressed model</text>
          <text x="810" y="623" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">• Run inference locally</text>

          <rect x="690" y="640" width="240" height="70" rx="6" fill="rgba(255,255,255,0.15)" />
          <text x="810" y="660" fontFamily="Arial, sans-serif" fontSize="13" fill="white" textAnchor="middle" fontWeight="bold">Data Collection</text>
          <text x="810" y="678" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">• Collect input-output pairs</text>
          <text x="810" y="693" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">• Upload to central server</text>
        </g>

        {/* Upload arrows back to server */}
        <path d="M 230 495 Q 150 400, 280 330"
              stroke="#8B5CF6"
              strokeWidth="2.5"
              fill="none"
              markerEnd="url(#cloudArrowPurple)"
              strokeDasharray="5,5" />
        <path d="M 770 495 Q 850 400, 720 330"
              stroke="#8B5CF6"
              strokeWidth="2.5"
              fill="none"
              markerEnd="url(#cloudArrowPurple)"
              strokeDasharray="5,5" />

        <text x="135" y="420" fontFamily="Arial, sans-serif" fontSize="13" fill="#8B5CF6" fontWeight="600">Data Upload</text>
        <text x="830" y="420" fontFamily="Arial, sans-serif" fontSize="13" fill="#8B5CF6" fontWeight="600">Data Upload</text>

        {/* Legend */}
        <g transform="translate(50, 745)">
          <text x="0" y="0" fontFamily="Arial, sans-serif" fontSize="12" fill="white" fontWeight="600">Data Flow:</text>
          <line x1="80" y1="-5" x2="120" y2="-5" stroke="#3B82F6" strokeWidth="2" />
          <text x="130" y="0" fontFamily="Arial, sans-serif" fontSize="11" fill="white">HTTP API</text>
          <line x1="190" y1="-5" x2="230" y2="-5" stroke="#10B981" strokeWidth="2" />
          <text x="240" y="0" fontFamily="Arial, sans-serif" fontSize="11" fill="white">Model Download</text>
          <line x1="345" y1="-5" x2="385" y2="-5" stroke="#8B5CF6" strokeWidth="2" strokeDasharray="5,5" />
          <text x="395" y="0" fontFamily="Arial, sans-serif" fontSize="11" fill="white">Data Upload</text>
        </g>
      </svg>
    </div>
  )
}
