import { useEffect, useRef, useState } from 'react'
import type { DetectedObject, ObjectDetection } from '@tensorflow-models/coco-ssd'
import {
  Activity, AlertTriangle, ArrowDownToLine, ArrowUpRight, Bell, Camera, CarFront,
  Check, ChevronDown, CircleHelp, Clock3, Cpu, Eye, Filter, LayoutDashboard,
  LockKeyhole, Radio, ScanEye, Search, ShieldCheck, SlidersHorizontal, Sparkles,
  UserRound, Video, Wifi, X, Zap,
} from 'lucide-react'
import './surveillance.css'

type ModelState = 'idle' | 'loading' | 'ready' | 'error'
type ActivityItem = { id: number; label: string; score: number; time: number }

const vehicleClasses = new Set(['bicycle', 'car', 'motorcycle', 'bus', 'truck'])

function currentTime() {
  return Date.now()
}

function clockTime(value: number) {
  return new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(value)
}

function App() {
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null)
  const [modelState, setModelState] = useState<ModelState>('idle')
  const [detections, setDetections] = useState<DetectedObject[]>([])
  const [activity, setActivity] = useState<ActivityItem[]>([])
  const [confidence, setConfidence] = useState(55)
  const [eventQuery, setEventQuery] = useState('')
  const [eventFilter, setEventFilter] = useState<'all' | 'person' | 'vehicle'>('all')
  const [connectionError, setConnectionError] = useState('')
  const [lastFrameAt, setLastFrameAt] = useState<number | null>(null)
  const [videoSize, setVideoSize] = useState({ width: 1, height: 1 })
  const confidenceRef = useRef(confidence)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const modelRef = useRef<ObjectDetection | null>(null)
  const inferenceTimerRef = useRef<number | null>(null)
  const inferenceActiveRef = useRef(false)
  const eventIdRef = useRef(0)
  const seenClassesRef = useRef(new Set<string>())

  const peopleCount = detections.filter((item) => item.class === 'person').length
  const vehicleCount = detections.filter((item) => vehicleClasses.has(item.class)).length
  const filteredActivity = activity.filter((item) => {
    const matchesType = eventFilter === 'all'
      || (eventFilter === 'person' && item.label === 'person')
      || (eventFilter === 'vehicle' && vehicleClasses.has(item.label))
    return matchesType && item.label.includes(eventQuery.trim().toLowerCase())
  })

  function disconnectCamera() {
    inferenceActiveRef.current = false
    if (inferenceTimerRef.current !== null) window.clearTimeout(inferenceTimerRef.current)
    inferenceTimerRef.current = null
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    setCameraStream(null)
    setDetections([])
    seenClassesRef.current.clear()
    setLastFrameAt(null)
    setModelState(modelRef.current ? 'ready' : 'idle')
    setConnectionError('')
  }

  async function detectFrame() {
    if (!inferenceActiveRef.current) return
    const video = videoRef.current
    const model = modelRef.current
    if (!video || video.readyState < 2 || !model) {
      inferenceTimerRef.current = window.setTimeout(() => void detectFrame(), 300)
      return
    }

    try {
      const results = await model.detect(video, 20, confidenceRef.current / 100)
      if (!inferenceActiveRef.current) return
      setDetections(results)
      setLastFrameAt(currentTime())
      const visibleClasses = new Set(results.filter((item) => item.score >= 0.7).map((item) => item.class))
      const newlySeen = results.filter((item) => item.score >= 0.7 && !seenClassesRef.current.has(item.class))
      if (newlySeen.length > 0) {
        const detectedAt = currentTime()
        const newEvents = newlySeen.map((item) => ({
          id: eventIdRef.current++,
          label: item.class,
          score: item.score,
          time: detectedAt,
        }))
        setActivity((current) => [...newEvents, ...current].slice(0, 30))
      }
      seenClassesRef.current = visibleClasses
    } catch {
      if (inferenceActiveRef.current) {
        setConnectionError('The video stream is active, but frame analysis stopped. Try reconnecting the camera.')
        setModelState('error')
        inferenceActiveRef.current = false
      }
    }

    if (inferenceActiveRef.current) {
      inferenceTimerRef.current = window.setTimeout(() => void detectFrame(), 420)
    }
  }

  async function connectCamera() {
    setConnectionError('')
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Camera access needs a secure browser context. Open this app on localhost or HTTPS.')
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
      })
      streamRef.current = stream
      setCameraStream(stream)
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setModelState('loading')
      const [{ default: cocoSsd }, tf] = await Promise.all([
        import('@tensorflow-models/coco-ssd'),
        import('@tensorflow/tfjs'),
      ])
      await tf.ready()
      if (!modelRef.current) modelRef.current = await cocoSsd.load({ base: 'lite_mobilenet_v2' })
      if (streamRef.current !== stream) return
      setModelState('ready')
      inferenceActiveRef.current = true
      void detectFrame()
    } catch (error) {
      disconnectCamera()
      setModelState('error')
      setConnectionError(error instanceof Error ? error.message : 'Unable to start the camera or AI model.')
    }
  }

  useEffect(() => () => {
    inferenceActiveRef.current = false
    if (inferenceTimerRef.current !== null) window.clearTimeout(inferenceTimerRef.current)
    streamRef.current?.getTracks().forEach((track) => track.stop())
  }, [])

  return (
    <div className="app-shell" id="overview">
      <aside className="sidebar">
        <a className="brand" href="#overview" aria-label="Sentinel overview">
          <span className="brand-mark"><ScanEye size={19} strokeWidth={2.1} /></span>
          <span className="brand-name">sentinel<span> / AI</span></span>
        </a>
        <div className="workspace-label">WORKSPACE</div>
        <button className="site-switcher" type="button">
          <span className="site-icon"><Eye size={16} /></span>
          <span className="site-copy"><strong>North campus</strong><small>Security workspace</small></span>
          <ChevronDown size={15} />
        </button>
        <div className="nav-label">MONITORING</div>
        <nav className="side-nav" aria-label="Main navigation">
          <a className="nav-item is-current" href="#overview"><LayoutDashboard size={17} /> Overview <span className="nav-count">01</span></a>
          <a className="nav-item" href="#camera-feed"><Video size={17} /> Camera feed <span className="nav-count">01</span></a>
          <a className="nav-item" href="#event-log"><Activity size={17} /> Event log {activity.length > 0 && <span className="nav-count">{activity.length}</span>}</a>
        </nav>
        <div className="sidebar-divider" />
        <div className="nav-label">CAMERA SOURCES <span className="source-total">01</span></div>
        <div className={`source-card ${cameraStream ? 'source-card-active' : ''}`}>
          <span className="source-dot" />
          <span className="source-copy"><strong>Local device</strong><small>{cameraStream ? 'Streaming now' : 'Not connected'}</small></span>
          <span className="source-menu">•••</span>
        </div>
        <div className="sidebar-spacer" />
        <div className="privacy-note"><LockKeyhole size={14} /><span>Video stays on this device</span></div>
        <div className="profile-row">
          <div className="profile-avatar">NS</div>
          <div className="profile-copy"><strong>North security</strong><small>Operator</small></div>
          <CircleHelp size={17} className="help-icon" />
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumbs"><span>North campus</span><span className="crumb-divider">/</span><strong>Overview</strong></div>
          <div className="topbar-actions">
            <span className="local-pill"><span /> LOCAL INFERENCE</span>
            <button className="icon-button notification-button" type="button" aria-label="Notifications"><Bell size={17} /><i /></button>
            <div className="topbar-avatar">NS</div>
          </div>
        </header>

        <div className="page-content">
          <section className="page-heading">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> SECURITY OPERATIONS <span className="eyebrow-date">{new Intl.DateTimeFormat(undefined, { weekday: 'short', month: 'short', day: 'numeric' }).format(new Date())}</span></div>
              <h1>Security overview</h1>
              <p className="page-subtitle">Live situational awareness for North campus.</p>
            </div>
            <div className="heading-actions">
              <button className="button button-secondary" type="button" onClick={() => setActivity([])} disabled={activity.length === 0}><ArrowDownToLine size={15} /> Clear events</button>
              <button className={`button ${cameraStream ? 'button-stop' : 'button-primary'}`} type="button" onClick={cameraStream ? disconnectCamera : connectCamera} disabled={modelState === 'loading' && !cameraStream}>
                {cameraStream ? <><X size={15} /> Disconnect</> : <><Camera size={15} /> Connect camera</>}
              </button>
            </div>
          </section>

          <section className="metric-grid" aria-label="Live metrics">
            <article className="metric-card metric-card-live">
              <div className="metric-top"><span>PEOPLE IN VIEW</span><span className="metric-symbol person-symbol"><UserRound size={15} /></span></div>
              <div className="metric-value">{peopleCount.toString().padStart(2, '0')}<span className="metric-unit"> persons</span></div>
              <div className="metric-foot"><span className={cameraStream ? 'metric-live-dot' : 'metric-idle-dot'} />{cameraStream ? 'Live detection' : 'Awaiting camera source'}</div>
            </article>
            <article className="metric-card">
              <div className="metric-top"><span>VEHICLES IN VIEW</span><span className="metric-symbol vehicle-symbol"><CarFront size={16} /></span></div>
              <div className="metric-value">{vehicleCount.toString().padStart(2, '0')}<span className="metric-unit"> vehicles</span></div>
              <div className="metric-foot"><span className="metric-muted-dot" />Current frame</div>
            </article>
            <article className="metric-card">
              <div className="metric-top"><span>CAMERA SOURCES</span><span className="metric-symbol camera-symbol"><Camera size={15} /></span></div>
              <div className="metric-value">{cameraStream ? '01' : '00'}<span className="metric-unit"> / 01</span></div>
              <div className="metric-foot"><span className={cameraStream ? 'metric-live-dot' : 'metric-idle-dot'} />{cameraStream ? 'One source online' : 'No sources connected'}</div>
            </article>
            <article className="metric-card">
              <div className="metric-top"><span>EVENTS DETECTED</span><span className="metric-symbol event-symbol"><Zap size={15} /></span></div>
              <div className="metric-value">{activity.length.toString().padStart(2, '0')}<span className="metric-unit"> this session</span></div>
              <div className="metric-foot"><span className="metric-muted-dot" />On-device analysis</div>
            </article>
          </section>

          <div className="content-grid">
            <section className="panel camera-panel" id="camera-feed">
              <div className="panel-heading">
                <div className="panel-title-group"><div className="panel-kicker"><span className={cameraStream ? 'status-led is-live' : 'status-led'} /> CAMERA 01</div><h2>Local device <span className="heading-location"><span>•</span> North campus</span></h2></div>
                <div className="camera-tools"><span className="resolution-tag">{cameraStream ? 'LIVE' : 'OFFLINE'}</span><button className="icon-button" type="button" aria-label="Camera settings"><SlidersHorizontal size={16} /></button><button className="icon-button" type="button" aria-label="More camera options"><span className="more-dots">•••</span></button></div>
              </div>
              <div className={`video-stage ${cameraStream ? 'video-stage-live' : ''}`}>
                <video ref={videoRef} autoPlay muted playsInline aria-label="Local camera feed" onLoadedMetadata={(event) => setVideoSize({ width: event.currentTarget.videoWidth, height: event.currentTarget.videoHeight })} />
                {!cameraStream && <div className="empty-camera">
                  <div className="camera-reticle"><span /><span /><span /><span /><Camera size={23} /></div>
                  <div className="empty-title">No camera connected</div><p>Connect a camera to begin local AI analysis.</p>
                  <button className="button button-light" type="button" onClick={connectCamera} disabled={modelState === 'loading'}><Camera size={15} /> Connect local camera</button>
                  <span className="feed-permission"><LockKeyhole size={12} /> Access is requested only when you connect</span>
                </div>}
                {cameraStream && detections.map((item, index) => {
                  const [left, top, boxWidth, boxHeight] = item.bbox
                  const color = item.class === 'person' ? 'mint' : vehicleClasses.has(item.class) ? 'amber' : 'ice'
                  return <div className={`detection-box detection-${color}`} key={`${item.class}-${index}`} style={{ left: `${left / videoSize.width * 100}%`, top: `${top / videoSize.height * 100}%`, width: `${boxWidth / videoSize.width * 100}%`, height: `${boxHeight / videoSize.height * 100}%` }}><span>{item.class} <b>{Math.round(item.score * 100)}%</b></span></div>
                })}
                {cameraStream && <div className="feed-live-label"><span /> LIVE</div>}
                <div className="feed-corner feed-corner-tl" /><div className="feed-corner feed-corner-br" />
              </div>
              <div className="feed-footer"><div className="feed-meta"><span><Radio size={14} /> {cameraStream ? 'SOURCE ACTIVE' : 'SOURCE IDLE'}</span><i />{lastFrameAt ? `Frame ${clockTime(lastFrameAt)}` : 'Waiting for input'}</div><div className="feed-quality"><span className={cameraStream ? 'quality-led quality-led-on' : 'quality-led'} />{cameraStream ? 'Analysis running' : 'No signal'}</div></div>
              {connectionError && <div className="inline-error"><AlertTriangle size={15} /> {connectionError}</div>}
            </section>

            <aside className="panel intelligence-panel">
              <div className="panel-heading intelligence-heading"><div><div className="panel-kicker"><Sparkles size={13} /> AI INTELLIGENCE</div><h2>Detection settings</h2></div><span className="model-badge">COCO-SSD</span></div>
              <div className="model-status-row"><span className={`model-icon ${modelState === 'ready' ? 'model-icon-ready' : ''}`}><Cpu size={17} /></span><div className="model-copy"><strong>Object detection</strong><small>{modelState === 'loading' ? 'Loading model weights…' : modelState === 'ready' ? 'Model ready · MobileNet v2' : modelState === 'error' ? 'Model unavailable' : 'Ready when camera connects'}</small></div><span className={`model-state-dot model-${modelState}`} /></div>
              <div className="settings-block"><div className="setting-label-row"><label htmlFor="confidence">Confidence threshold</label><strong>{confidence}%</strong></div><input id="confidence" type="range" min="30" max="90" step="5" value={confidence} onChange={(event) => { const value = Number(event.target.value); confidenceRef.current = value; setConfidence(value) }} /><div className="range-labels"><span>More detections</span><span>Higher certainty</span></div></div>
              <div className="class-list-heading"><span>TRACKING CLASSES</span><span>ENABLED</span></div>
              <div className="class-row"><span className="class-icon class-person"><UserRound size={14} /></span><span className="class-name">Person</span><Check size={15} className="class-check" /></div>
              <div className="class-row"><span className="class-icon class-vehicle"><CarFront size={15} /></span><span className="class-name">Vehicles</span><Check size={15} className="class-check" /></div>
              <div className="class-row"><span className="class-icon class-object"><Eye size={14} /></span><span className="class-name">Common objects</span><Check size={15} className="class-check" /></div>
              <div className="privacy-callout"><LockKeyhole size={15} /><span><strong>Private by design</strong><small>Frames are analyzed locally and never recorded or uploaded.</small></span></div>
              <div className="model-footnote"><Wifi size={13} /> Model weights download from TensorFlow Hub on first use.</div>
            </aside>
          </div>

          <section className="panel events-panel" id="event-log">
            <div className="events-header"><div><div className="panel-kicker"><Activity size={13} /> LIVE ACTIVITY</div><h2>Detection events <span className="event-total">{activity.length.toString().padStart(2, '0')}</span></h2></div>
              <div className="event-controls"><label className="event-search"><Search size={15} /><input aria-label="Search events" placeholder="Search events" value={eventQuery} onChange={(event) => setEventQuery(event.target.value)} /><kbd>/</kbd></label><div className="filter-control"><Filter size={14} /><select aria-label="Filter events" value={eventFilter} onChange={(event) => setEventFilter(event.target.value as typeof eventFilter)}><option value="all">All events</option><option value="person">People</option><option value="vehicle">Vehicles</option></select><ChevronDown size={13} /></div></div>
            </div>
            <div className="event-table-head"><span>DETECTION</span><span>CAMERA</span><span>CONFIDENCE</span><span>TIME</span><span /></div>
            <div className="event-list">{filteredActivity.length === 0 ? <div className="empty-events"><span className="empty-event-icon"><ScanEye size={18} /></span><span><strong>{activity.length === 0 ? 'No detections recorded' : 'No matching events'}</strong><small>{activity.length === 0 ? 'Detected objects will appear here while the camera is connected.' : 'Try another search or event filter.'}</small></span></div> : filteredActivity.map((item) => <div className="event-row" key={item.id}>
              <span className="event-detection"><span className={`event-object-icon ${item.label === 'person' ? 'object-person' : 'object-vehicle'}`}>{item.label === 'person' ? <UserRound size={15} /> : <CarFront size={15} />}</span><span><strong>{item.label[0].toUpperCase() + item.label.slice(1)} detected</strong><small>Motion in camera frame</small></span></span>
              <span className="event-camera"><Camera size={14} /> Local device</span><span className="event-confidence"><span>{Math.round(item.score * 100)}%</span><i><b style={{ width: `${Math.round(item.score * 100)}%` }} /></i></span><span className="event-time"><Clock3 size={13} /> {clockTime(item.time)}</span><span className="event-action"><ArrowUpRight size={15} /></span>
            </div>)}</div>
            <div className="events-footer"><span><span className="footer-pulse" /> Updates as objects enter the frame</span><button type="button" className="text-button" onClick={() => document.getElementById('camera-feed')?.scrollIntoView({ behavior: 'smooth', block: 'center' })}>View camera <ArrowUpRight size={13} /></button></div>
          </section>
          <footer className="page-footer"><span><ShieldCheck size={14} /> Sentinel local monitoring</span><span><span className="footer-pulse" /> SYSTEM OPERATIONAL</span><span>AI-assisted detection · Verify alerts before acting</span></footer>
        </div>
      </main>
    </div>
  )
}

export default App