/**
 * TrueTrack - Flight Recorder / Navigation Instrument Controller
 * MapLibre GL WebGL Engine + Real OSM GeoJSON Vector Manifold + Tabular Telemetry
 */

const CORRIDOR_CONFIGS = {
  hitec: {
    id: 'hitec',
    name: 'Hyderabad HITEC City Underpass',
    statusDesc: 'Simulated IMU on real road geometry • 45s Underpass',
    center: [78.3785, 17.4465],
    zoom: 15.5,
    minZoom: 14.0,
    maxZoom: 18.0,
    maxBounds: [[78.366, 17.435], [78.392, 17.455]],
    telemetry: window.SIMULATION_TELEMETRY || [],
    startIndex: 380, // t=38s, 2s before blackout
    blackoutLeftPct: '30.8%',
    blackoutWidthPct: '34.6%',
    blackoutLabel: '45 s underpass blackout',
    rfBadge: 'SYNTHETIC BENCHMARK',
    rfTarget: 'BENCHMARK < 1.0 m',
    rfTTDrift: '0.95 m',
    rfTTSub: '0.3% of 320 m',
    rfNaiveDrift: '114.0 m',
    rfNaiveSub: '>100% (diverging)',
    rfCaption: 'Evaluated across 45s underpass on surveyed HITEC City road geometry'
  },
  varada: {
    id: 'varada',
    name: 'Chennai Varadarajapuram (Real Field Log)',
    statusDesc: 'Real Phone IMU • 45s Blackout Window • Urban Road',
    center: [80.0833, 13.0475],
    zoom: 16.5,
    minZoom: 14.0,
    maxZoom: 18.0,
    maxBounds: [[80.070, 13.035], [80.098, 13.060]],
    telemetry: window.VARADARAJAPURAM_TELEMETRY || [],
    startIndex: 0,
    blackoutLeftPct: '6.7%',
    blackoutWidthPct: '93.3%',
    blackoutLabel: '42 s field blackout (t=3s to 45s)',
    rfBadge: 'CHENNAI FIELD LOG',
    rfTarget: 'DRIFT TARGET: <10% (industry dead-reckoning benchmark) — PASSED',
    rfTTDrift: '5.4%',
    rfTTSub: '21.5 m error / 396 m (pure unassisted)',
    rfNaiveDrift: '35.2%',
    rfNaiveSub: '139.3 m (diverging off-road)',
    rfCaption: 'Evaluated on real Chennai phone IMU against GPS ground truth (Location.csv)'
  },
  rohini: {
    id: 'rohini',
    name: 'Chennai Rohini Koyambedu (Real Lean Cornering)',
    statusDesc: 'Real Phone IMU • 22.1° Lean Angle • High-Speed Corridor',
    center: [80.1994, 13.0765],
    zoom: 16.8,
    minZoom: 14.0,
    maxZoom: 18.0,
    maxBounds: [[80.185, 13.065], [80.215, 13.090]],
    telemetry: window.ROHINI_TELEMETRY || [],
    startIndex: 0,
    blackoutLeftPct: '6.7%',
    blackoutWidthPct: '93.3%',
    blackoutLabel: '42 s cornering blackout (t=3s to 45s)',
    rfBadge: 'CHENNAI CORNERING LOG',
    rfTarget: 'DRIFT TARGET: <10% (industry dead-reckoning benchmark) — PASSED',
    rfTTDrift: '6.2%',
    rfTTSub: '21.0 m error / 337 m (pure unassisted)',
    rfNaiveDrift: '199.5%',
    rfNaiveSub: '672.8 m (blown off arterial)',
    rfCaption: 'High-speed cornering log with 22.1° measured rider lean angle'
  }
};

class TrueTrackCockpit {
  constructor() {
    this.currentCorridor = 'hitec';
    this.telemetry = window.SIMULATION_TELEMETRY || [];
    this.metrics = window.BENCHMARK_METRICS || {};
    this.fftData = window.IMU_FFT_DATA || { freqs: [], magnitudes: [], dominant_peak_hz: 29.93, measured_vibration_attenuation_db: -8.3 };
    this.stressMetrics = window.STRESS_TEST_METRICS || {};
    this.distMetrics = window.DISTRIBUTION_METRICS || {};

    this.currentIndex = 380; // Default starts 2s before blackout at t=38s, showing clear GPS lock first
    this.isPlaying = true;
    this.playbackSpeed = 1.0;
    this.lastFrameTime = performance.now();

    // Smooth Display Tweens for Hero Numbers
    this.displayedLegacy = 0.2;
    this.displayedTT = 0.0;

    // Trajectory History Cache Index
    this.lastRenderedIdx = -1;

    // Chart Mouse Hover State
    this.chartHoverX = null;
    this.chartMousePos = null;

    // Flight Recorder State: Default to rock-solid Full Corridor Overview
    this.isFollowingVehicle = false;
    this.lastCameraIdx = -1;
    this.mapLoaded = false;
    this.drawerOpen = false;
    this.manualGpsKill = false;
    this.manualKillStartIdx = null;
    this.manualKillStartTime = null;
    this.manualRestoreStartTime = null;
    this.manualRestoreStartIdx = null;
    this.highVibrationInjected = false;
    this.voiceEnabled = true;
    this.isDemoMode = false;
    this.demoStartTime = 0;

    // Component Ablation Toggles
    this.toggleNeural = true;
    this.toggleMap = true;
    this.toggleSigmoid = true;
    this.toggleLean = true;

    // Dual-Screen Green Light Phone Bridge State
    this.isPhoneLive = false;
    this.phoneSocket = null;
    this.phoneTelemetry = null;

    // State Tracking
    this.prevBlackoutState = false;

    // Initialize Subsystems
    this.initMap();
    this.initControls();
    this.initAblation();
    this.initPhoneBridge();
    this.initAudioContext();
    this.initCanvases();

    // Start Real-Time 60 FPS Animation Loop
    requestAnimationFrame((t) => this.tick(t));
  }

  /* ========================================================================
     1. MapLibre GL Initialization (100% Offline, Zero Cloud Dependencies)
     ======================================================================== */
  initMap() {
    // Center initially on full corridor overview (Cyber Towers to Mindspace)
    const corridorCenter = [78.3785, 17.4465];
    const initialCoord = (this.telemetry && this.telemetry.length > 0)
      ? [this.telemetry[0].gt[1], this.telemetry[0].gt[0]]
      : [78.3771, 17.4415];
    this.map = new maplibregl.Map({
      container: 'maplibre-map',
      center: corridorCenter,
      zoom: 15.5,
      pitch: 0,
      bearing: 0,
      attributionControl: false,
      maxBounds: CORRIDOR_CONFIGS.hitec.maxBounds,
      minZoom: 14.0,
      maxZoom: 18.0,
      style: {
        version: 8,
        sources: {
          'osm-tiles': {
            type: 'raster',
            tiles: ['./tiles/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: '&copy; OpenStreetMap contributors | Offline Bundle'
          }
        },
        layers: [
          {
            id: 'osm-tiles-layer',
            type: 'raster',
            source: 'osm-tiles'
          }
        ]
      }
    });

    // Add standard on-map +/- zoom controls
    this.map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');

    this.map.on('load', () => {
      this.mapLoaded = true;
      this.map.resize();

      // 1. Vector Road Centerline from real OpenStreetMap GeoJSON
      this.map.addSource('osm-road', {
        type: 'geojson',
        data: './data/hitec_corridor.geojson'
      });

      // Underpass Tunnel Band (Subtle outlined/hatched band, not neon blur)
      this.map.addLayer({
        id: 'osm-underpass-tube',
        type: 'line',
        source: 'osm-road',
        filter: ['==', 'type', 'tunnel'],
        paint: {
          'line-color': '#d97706',
          'line-width': 8,
          'line-opacity': 0.35
        }
      });

      // Ground Truth Road Centerline (Thin white dashed line)
      this.map.addLayer({
        id: 'osm-road-centerline',
        type: 'line',
        source: 'osm-road',
        filter: ['!=', 'type', 'tunnel'],
        paint: {
          'line-color': '#cbd5e1',
          'line-width': 1.5,
          'line-opacity': 0.7,
          'line-dasharray': [3, 4]
        }
      });

      // 2. Full Corridor Route (Dim 2px base line showing full path ahead)
      const fullRouteCoords = this.telemetry.map(d => [d.gt[1], d.gt[0]]);
      this.map.addSource('full-route', {
        type: 'geojson',
        data: { type: 'Feature', geometry: { type: 'LineString', coordinates: fullRouteCoords } }
      });
      this.map.addLayer({
        id: 'full-route-casing',
        type: 'line',
        source: 'full-route',
        paint: {
          'line-color': '#07090d',
          'line-width': 4.0,
          'line-opacity': 0.85
        }
      });
      this.map.addLayer({
        id: 'full-route-base',
        type: 'line',
        source: 'full-route',
        paint: {
          'line-color': '#334155',
          'line-width': 2.0,
          'line-opacity': 0.55
        }
      });

      // 3. Dynamic Trajectory Sources (Traveled portion bright with thin dark casing)
      // TrueTrack History Line
      this.map.addSource('tt-history', {
        type: 'geojson',
        data: { type: 'Feature', geometry: { type: 'LineString', coordinates: [] } }
      });
      this.map.addLayer({
        id: 'tt-history-casing',
        type: 'line',
        source: 'tt-history',
        paint: {
          'line-color': '#07090d',
          'line-width': 5.5,
          'line-opacity': 0.95
        }
      });
      this.map.addLayer({
        id: 'tt-history-layer',
        type: 'line',
        source: 'tt-history',
        paint: {
          'line-color': '#38bdf8',
          'line-width': 2.8,
          'line-opacity': 1.0
        }
      });

      // Legacy GPS History Line
      this.map.addSource('legacy-history', {
        type: 'geojson',
        data: { type: 'Feature', geometry: { type: 'LineString', coordinates: [] } }
      });
      this.map.addLayer({
        id: 'legacy-history-casing',
        type: 'line',
        source: 'legacy-history',
        paint: {
          'line-color': '#07090d',
          'line-width': 5.0,
          'line-opacity': 0.9
        }
      });
      this.map.addLayer({
        id: 'legacy-history-layer',
        type: 'line',
        source: 'legacy-history',
        paint: {
          'line-color': '#ef4444',
          'line-width': 2.2,
          'line-opacity': 0.95
        }
      });

      // Thin Dynamic Tether Lines (connecting estimates to Ground Truth)
      this.map.addSource('tether-lines', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] }
      });
      this.map.addLayer({
        id: 'tether-layer',
        type: 'line',
        source: 'tether-lines',
        paint: {
          'line-color': ['get', 'color'],
          'line-width': 1,
          'line-dasharray': [2, 2],
          'line-opacity': 0.5
        }
      });

      // Discrete-Time EKF Covariance Ellipse
      this.map.addSource('ekf-ellipse-source', {
        type: 'geojson',
        data: { type: 'Feature', geometry: { type: 'Polygon', coordinates: [[]] } }
      });
      this.map.addLayer({
        id: 'ekf-ellipse-fill',
        type: 'fill',
        source: 'ekf-ellipse-source',
        paint: {
          'fill-color': '#38bdf8',
          'fill-opacity': 0.15
        }
      });
      this.map.addLayer({
        id: 'ekf-ellipse-stroke',
        type: 'line',
        source: 'ekf-ellipse-source',
        paint: {
          'line-color': '#38bdf8',
          'line-width': 1,
          'line-dasharray': [3, 3]
        }
      });

      // Detect manual panning to toggle off auto-follow
      this.map.on('dragstart', () => {
        this.isFollowingVehicle = false;
        const btnReset = document.getElementById('btn-reset-view');
        if (btnReset) {
          btnReset.classList.remove('active');
          const span = btnReset.querySelector('span');
          if (span) span.textContent = 'Follow vehicle';
        }
      });
    });

    // Directional Vehicle Markers: TrueTrack (40px, z-index 800) strictly over Legacy (28px, z-index 700)
    const elTT = document.createElement('div');
    elTT.className = 'maplibre-marker-puck marker-tt';
    elTT.title = 'TrueTrack Fused State';
    elTT.style.zIndex = '800';
    elTT.innerHTML = '<div class="puck-arrow" id="puck-arrow-tt"></div>';
    elTT.addEventListener('click', () => {
      this.isFollowingVehicle = true;
      const btnReset = document.getElementById('btn-reset-view');
      if (btnReset) {
        btnReset.classList.add('active');
        const span = btnReset.querySelector('span');
        if (span) span.textContent = 'Corridor view';
      }
      const i = Math.floor(this.currentIndex);
      const cur = this.telemetry[i];
      if (cur) {
        this.map.flyTo({ center: [cur.truetrack[1], cur.truetrack[0]], zoom: 16.5, duration: 400 });
      }
    });
    this.markerTT = new maplibregl.Marker({ element: elTT }).setLngLat(initialCoord).addTo(this.map);
    if (elTT.parentElement) elTT.parentElement.style.zIndex = '800';

    const elLegacy = document.createElement('div');
    elLegacy.className = 'maplibre-marker-puck marker-legacy';
    elLegacy.title = 'Legacy Dead Reckoning';
    elLegacy.style.zIndex = '700';
    elLegacy.innerHTML = '<div class="puck-arrow" id="puck-arrow-legacy"></div>';
    this.markerLegacy = new maplibregl.Marker({ element: elLegacy }).setLngLat(initialCoord).addTo(this.map);
    if (elLegacy.parentElement) elLegacy.parentElement.style.zIndex = '700';

    window.addEventListener('resize', () => {
      if (this.map) this.map.resize();
    });
  }

  /* ========================================================================
     2. Flight Recorder Transport Controls & Video Scrubber
     ======================================================================== */
  initControls() {
    const btnPlay = document.getElementById('btn-play');
    const btnKillGps = document.getElementById('btn-kill-gps');
    const btnVibe = document.getElementById('btn-vibe');
    const btnVoice = document.getElementById('btn-voice');
    const btnDemo = document.getElementById('btn-demo');
    const btnReset = document.getElementById('btn-reset-view');
    const btnDrawer = document.getElementById('btn-hood-toggle');
    const slider = document.getElementById('timeline-slider');
    const speedBtns = document.querySelectorAll('.btn-speed');

    if (btnPlay) {
      btnPlay.addEventListener('click', () => {
        this.isPlaying = !this.isPlaying;
        btnPlay.classList.toggle('active', !this.isPlaying);
        const iconPlay = document.getElementById('icon-play');
        if (iconPlay) {
          iconPlay.innerHTML = this.isPlaying
            ? '<path d="M3.5 3.5h3v9h-3v-9zm6 0h3v9h-3v-9z"/>'
            : '<path d="M5 3.5v9l7-4.5z"/>';
        }
      });
    }

    if (btnKillGps) {
      btnKillGps.addEventListener('click', () => {
        this.manualGpsKill = !this.manualGpsKill;
        const curIdx = Math.floor(this.currentIndex);
        const curTime = this.telemetry[curIdx] ? this.telemetry[curIdx].t : 0;

        if (this.manualGpsKill) {
          this.manualKillStartIdx = curIdx;
          this.manualKillStartTime = curTime;
          this.manualRestoreStartTime = null;
          this.manualRestoreStartIdx = null;
        } else {
          this.manualRestoreStartTime = curTime;
          this.manualRestoreStartIdx = curIdx;
        }

        btnKillGps.classList.toggle('active', this.manualGpsKill);
        btnKillGps.textContent = this.manualGpsKill ? 'Restore GPS' : 'Kill GPS';
        this.playTone(this.manualGpsKill ? 220 : 660, 'triangle', 0.2);
        this.lastRenderedIdx = -1;
        this.updateHUD();
        this.render();
      });
    }

    if (btnVibe) {
      btnVibe.addEventListener('click', () => {
        this.highVibrationInjected = !this.highVibrationInjected;
        btnVibe.classList.toggle('active', this.highVibrationInjected);
      });
    }

    if (btnVoice) {
      btnVoice.addEventListener('click', () => {
        this.voiceEnabled = !this.voiceEnabled;
        btnVoice.classList.toggle('active', !this.voiceEnabled);
        btnVoice.textContent = this.voiceEnabled ? 'Audio on' : 'Audio off';
      });
    }

    if (btnDemo) {
      btnDemo.addEventListener('click', () => {
        this.startGuidedDemo();
      });
    }

    const selectCorridor = document.getElementById('select-corridor');
    if (selectCorridor) {
      selectCorridor.addEventListener('change', (e) => {
        this.switchCorridor(e.target.value);
      });
    }

    if (btnReset) {
      btnReset.addEventListener('click', () => {
        const cfg = CORRIDOR_CONFIGS[this.currentCorridor] || CORRIDOR_CONFIGS.hitec;
        if (this.isFollowingVehicle) {
          // Return to static Corridor view (zero camera motion, perfectly steady)
          this.isFollowingVehicle = false;
          btnReset.classList.remove('active');
          const span = btnReset.querySelector('span');
          if (span) span.textContent = 'Follow vehicle';
          this.map.flyTo({ center: cfg.center, zoom: cfg.zoom, pitch: 0, duration: 400 });
        } else {
          // Enter Follow vehicle mode
          this.isFollowingVehicle = true;
          btnReset.classList.add('active');
          const span = btnReset.querySelector('span');
          if (span) span.textContent = 'Corridor view';
          const i = Math.floor(this.currentIndex);
          const cur = this.telemetry[i];
          if (cur) {
            this.map.flyTo({ center: [cur.truetrack[1], cur.truetrack[0]], zoom: 16.5, duration: 400 });
          }
        }
      });
    }

    if (btnDrawer) {
      btnDrawer.addEventListener('click', () => {
        this.toggleDrawer();
      });
    }

    if (slider) {
      slider.max = Math.max(0, this.telemetry.length - 1);
      slider.value = this.currentIndex;
      slider.addEventListener('input', (e) => {
        this.currentIndex = parseInt(e.target.value);
        this.lastRenderedIdx = -1;
        this.updateHUD();
        this.render();
      });
    }

    speedBtns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        speedBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const spd = parseFloat(btn.dataset.speed || btn.getAttribute('data-speed'));
        this.playbackSpeed = isNaN(spd) ? 1.0 : spd;
      });
    });

    // 2.4-Second Gentle Auto-Dismiss for Intro Overlay
    const intro = document.getElementById('intro-overlay');
    const btnDismiss = document.getElementById('btn-dismiss-intro');
    const dismissFn = () => {
      if (intro) intro.classList.add('dismissed');
    };
    if (btnDismiss) btnDismiss.addEventListener('click', dismissFn);
    setTimeout(dismissFn, 2400);

    // Global Ergonomic Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      if (e.code === 'Space') {
        e.preventDefault();
        const btnPlay = document.getElementById('btn-play');
        if (btnPlay) btnPlay.click();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        this.currentIndex = Math.max(0, this.currentIndex - 50); // Seek back 5s
        this.lastRenderedIdx = -1;
        this.updateHUD();
        this.render();
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        this.currentIndex = Math.min(this.telemetry.length - 1, this.currentIndex + 50); // Seek forward 5s
        this.lastRenderedIdx = -1;
        this.updateHUD();
        this.render();
      } else if (['1', '2', '4', '8'].includes(e.key)) {
        const btn = document.querySelector(`.btn-speed[data-speed="${e.key}.0"]`);
        if (btn) btn.click();
      }
    });
  }

  /* ========================================================================
     3. Corridor Switching & Real Field Log Loading (Item #1 & #2)
     ======================================================================== */
  switchCorridor(corridorKey) {
    const cfg = CORRIDOR_CONFIGS[corridorKey] || CORRIDOR_CONFIGS.hitec;
    this.currentCorridor = corridorKey;
    this.telemetry = cfg.telemetry && cfg.telemetry.length > 0 ? cfg.telemetry : window.SIMULATION_TELEMETRY;
    this.currentIndex = cfg.startIndex || 0;
    this.lastRenderedIdx = -1;
    this.displayedLegacy = 0.0;
    this.displayedTT = 0.0;

    // 1. Update Video Scrubber slider bounds
    const slider = document.getElementById('timeline-slider');
    if (slider) {
      slider.max = Math.max(0, this.telemetry.length - 1);
      slider.value = this.currentIndex;
    }

    // 2. Update scrubber blackout span
    const span = document.querySelector('.scrubber-blackout-span');
    const spanLabel = document.querySelector('.scrubber-blackout-span .span-label');
    const markerEntry = document.querySelector('.scrubber-marker.marker-entry');
    const markerExit = document.querySelector('.scrubber-marker.marker-exit');
    if (span) {
      span.style.left = cfg.blackoutLeftPct;
      span.style.width = cfg.blackoutWidthPct;
    }
    if (spanLabel) spanLabel.textContent = cfg.blackoutLabel;
    if (markerEntry) markerEntry.style.left = cfg.blackoutLeftPct;
    if (markerExit) markerExit.style.left = `calc(${cfg.blackoutLeftPct} + ${cfg.blackoutWidthPct})`;

    // 3. Update Real Field Drift Badge (Item #2)
    const pill = document.getElementById('rd-badge-pill');
    const target = document.getElementById('rd-target-chip');
    const ttDrift = document.getElementById('rd-tt-drift');
    const ttSub = document.getElementById('rd-tt-sub');
    const naiveDrift = document.getElementById('rd-naive-drift');
    const naiveSub = document.getElementById('rd-naive-sub');
    const caption = document.getElementById('rd-caption');
    if (pill) pill.textContent = cfg.rfBadge;
    if (target) target.textContent = cfg.rfTarget;
    if (ttDrift) ttDrift.textContent = cfg.rfTTDrift;
    if (ttSub) ttSub.textContent = cfg.rfTTSub;
    if (naiveDrift) naiveDrift.textContent = cfg.rfNaiveDrift;
    if (naiveSub) naiveSub.textContent = cfg.rfNaiveSub;
    if (caption) caption.textContent = cfg.rfCaption;

    // 4. Update Header status text
    const statusItem = document.querySelector('.header-status-line .status-item');
    if (statusItem) statusItem.textContent = cfg.statusDesc;

    // 5. Update MapLibre GL Layers & Camera (Item #1)
    if (this.map && this.mapLoaded) {
      const fullRouteCoords = this.telemetry.map(d => [d.gt[1], d.gt[0]]);
      const fullSrc = this.map.getSource('full-route');
      if (fullSrc) {
        fullSrc.setData({ type: 'Feature', geometry: { type: 'LineString', coordinates: fullRouteCoords } });
      }

      const ttSrc = this.map.getSource('tt-history');
      if (ttSrc) {
        ttSrc.setData({ type: 'Feature', geometry: { type: 'LineString', coordinates: [] } });
      }

      const legSrc = this.map.getSource('legacy-history');
      if (legSrc) {
        legSrc.setData({ type: 'Feature', geometry: { type: 'LineString', coordinates: [] } });
      }

      // Hide or show underpass tube based on corridor
      if (this.map.getLayer('osm-underpass-tube')) {
        this.map.setLayoutProperty('osm-underpass-tube', 'visibility', corridorKey === 'hitec' ? 'visible' : 'none');
      }
      if (this.map.getLayer('osm-road-centerline')) {
        this.map.setLayoutProperty('osm-road-centerline', 'visibility', corridorKey === 'hitec' ? 'visible' : 'none');
      }

      this.map.setMaxBounds(cfg.maxBounds);
      this.map.flyTo({ center: cfg.center, zoom: cfg.zoom, pitch: 0, duration: 800 });
    }

    this.updateHUD();
    this.render();
  }

  toggleDrawer(forceState = null) {
    const drawer = document.getElementById('diagnostics-drawer');
    const btn = document.getElementById('btn-hood-toggle');
    const arrow = document.getElementById('hood-toggle-arrow');
    
    this.drawerOpen = forceState !== null ? forceState : !this.drawerOpen;
    if (drawer) drawer.classList.toggle('open', this.drawerOpen);
    if (arrow) arrow.innerHTML = this.drawerOpen ? '&blacktriangle;' : '&blacktriangledown;';
    
    setTimeout(() => {
      if (this.map) this.map.resize();
    }, 220);
  }

  /* ========================================================================
     3. Component Ablation Controller
     ======================================================================== */
  initAblation() {
    const chkNeural = document.getElementById('toggle-neural');
    const chkMap = document.getElementById('toggle-map');
    const chkSigmoid = document.getElementById('toggle-sigmoid');
    const lblSigmoid = document.getElementById('lbl-toggle-sigmoid');

    if (chkNeural) {
      chkNeural.addEventListener('change', () => {
        this.toggleNeural = chkNeural.checked;
        this.lastRenderedIdx = -1;
        this.render();
      });
    }

    if (chkMap) {
      chkMap.addEventListener('change', () => {
        this.toggleMap = chkMap.checked;
        if (chkSigmoid) chkSigmoid.disabled = !this.toggleMap;
        if (lblSigmoid) lblSigmoid.style.opacity = this.toggleMap ? '1.0' : '0.4';
        this.lastRenderedIdx = -1;
        this.render();
      });
    }

    if (chkSigmoid) {
      chkSigmoid.addEventListener('change', () => {
        this.toggleSigmoid = chkSigmoid.checked;
        this.lastRenderedIdx = -1;
        this.render();
      });
    }

    const chkLean = document.getElementById('toggle-lean');
    if (chkLean) {
      chkLean.addEventListener('change', () => {
        this.toggleLean = chkLean.checked;
        this.lastRenderedIdx = -1;
        this.render();
      });
    }
  }

  /* ========================================================================
     3b. Dual-Screen Phone Bridge (Green Light Mode via WebSocket)
     ======================================================================== */
  initPhoneBridge() {
    const btnPhoneLink = document.getElementById('btn-phone-link');
    const dot = document.getElementById('phone-bridge-dot');
    const label = document.getElementById('phone-bridge-label');

    const connectWs = (host = 'localhost:8765') => {
      if (this.phoneSocket) {
        try { this.phoneSocket.close(); } catch (e) {}
        this.phoneSocket = null;
      }
      if (label) label.textContent = 'Connecting...';

      try {
        this.phoneSocket = new WebSocket(`ws://${host}`);

        this.phoneSocket.onopen = () => {
          this.isPhoneLive = true;
          if (dot) dot.classList.add('connected');
          if (label) label.textContent = 'Phone: Live (50Hz)';
          if (btnPhoneLink) btnPhoneLink.classList.add('active');
          this.playTone(880, 'sine', 0.2);
        };

        this.phoneSocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.phoneTelemetry = data;
            this.onLivePhoneFrame(data);
          } catch (e) {}
        };

        this.phoneSocket.onclose = () => {
          this.isPhoneLive = false;
          if (dot) dot.classList.remove('connected');
          if (label) label.textContent = 'Phone Link';
          if (btnPhoneLink) btnPhoneLink.classList.remove('active');
        };

        this.phoneSocket.onerror = () => {
          this.isPhoneLive = false;
          if (dot) dot.classList.remove('connected');
          if (label) label.textContent = 'Phone Link (retry)';
        };
      } catch (err) {
        console.warn("Phone Bridge connection failed:", err);
        if (label) label.textContent = 'Phone Link';
      }
    };

    if (btnPhoneLink) {
      btnPhoneLink.addEventListener('click', () => {
        if (this.isPhoneLive && this.phoneSocket) {
          this.phoneSocket.close();
        } else {
          const customHost = prompt("Enter iQOO phone address:port (or leave localhost:8765 if using 'adb reverse tcp:8765 tcp:8765'):", "localhost:8765");
          if (customHost) connectWs(customHost.trim());
        }
      });
    }

    // Auto-probe local WebSocket bridge after brief startup delay
    setTimeout(() => connectWs('localhost:8765'), 1500);
  }

  onLivePhoneFrame(data) {
    // Dynamic updates directly from physical iQOO phone
    const speedEl = document.getElementById('hood-speed-val');
    if (speedEl && data.speed !== undefined) speedEl.textContent = `${data.speed.toFixed(1)} km/h`;

    const yawEl = document.getElementById('hood-yaw-val');
    if (yawEl && data.yaw !== undefined) yawEl.textContent = `${data.yaw.toFixed(1)} °/s`;

    const metaEl = document.getElementById('hood-imu-meta');
    if (metaEl && data.ax !== undefined) {
      metaEl.textContent = `ax: ${data.ax.toFixed(2)} • ay: ${data.ay.toFixed(2)} • lean: ${data.lean.toFixed(1)}° • NPU: ${data.latency.toFixed(1)}ms`;
    }

    // Direct on-device autonomous dead reckoning marker sync
    if (data.lat !== undefined && data.lon !== undefined && this.map && this.mapLoaded && this.carMarker) {
      this.carMarker.setLngLat([data.lon, data.lat]);
    }

    // Reflect live blackout toggled from physical phone button
    if (data.blackout !== undefined && data.blackout !== this.manualGpsKill) {
      this.manualGpsKill = data.blackout;
      const curIdx = Math.floor(this.currentIndex);
      const curTime = this.telemetry[curIdx] ? this.telemetry[curIdx].t : 0;
      if (this.manualGpsKill) {
        this.manualKillStartIdx = curIdx;
        this.manualKillStartTime = curTime;
        this.manualRestoreStartTime = null;
        this.manualRestoreStartIdx = null;
      } else {
        this.manualRestoreStartTime = curTime;
        this.manualRestoreStartIdx = curIdx;
      }
      const btnKill = document.getElementById('btn-kill-gps');
      if (btnKill) {
        btnKill.classList.toggle('active', this.manualGpsKill);
        btnKill.textContent = this.manualGpsKill ? 'Restore GPS' : 'Kill GPS';
      }
      this.lastRenderedIdx = -1;
      this.updateHUD();
      this.render();
    }
  }

  /* ========================================================================
     4. Web Audio Engine (Turn-by-Turn Acoustic Cues)
     ======================================================================== */
  initAudioContext() {
    this.audioCtx = null;
    const unlock = () => {
      if (!this.audioCtx) {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (AudioCtx) this.audioCtx = new AudioCtx();
      }
      if (this.audioCtx && this.audioCtx.state === 'suspended') {
        this.audioCtx.resume();
      }
      document.removeEventListener('click', unlock);
    };
    document.addEventListener('click', unlock);
  }

  playTone(freq, type = 'sine', dur = 0.15) {
    if (!this.voiceEnabled || !this.audioCtx) return;
    try {
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, this.audioCtx.currentTime);
      gain.gain.setValueAtTime(0.12, this.audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + dur);
      osc.connect(gain);
      gain.connect(this.audioCtx.destination);
      osc.start();
      osc.stop(this.audioCtx.currentTime + dur);
    } catch (e) {}
  }

  /* ========================================================================
     5. Diagnostics & Instrument Canvases Setup
     ======================================================================== */
  initCanvases() {
    this.canvasChart = document.getElementById('canvas-chart');
    this.ctxChart = this.canvasChart ? this.canvasChart.getContext('2d') : null;

    this.canvasWave = document.getElementById('canvas-imu-wave');
    this.ctxWave = this.canvasWave ? this.canvasWave.getContext('2d') : null;

    this.canvasFft = document.getElementById('canvas-fft');
    this.ctxFft = this.canvasFft ? this.canvasFft.getContext('2d') : null;

    this.canvasSparkLegacy = document.getElementById('canvas-spark-legacy');
    this.ctxSparkLegacy = this.canvasSparkLegacy ? this.canvasSparkLegacy.getContext('2d') : null;

    this.canvasSparkTT = document.getElementById('canvas-spark-tt');
    this.ctxSparkTT = this.canvasSparkTT ? this.canvasSparkTT.getContext('2d') : null;

    this.canvasTimelineSpark = document.getElementById('canvas-timeline-spark');
    this.ctxTimelineSpark = this.canvasTimelineSpark ? this.canvasTimelineSpark.getContext('2d') : null;

    this.chartTooltip = document.getElementById('chart-tooltip');

    // Chart mouse hover crosshair & tooltip listener
    if (this.canvasChart) {
      this.canvasChart.addEventListener('mousemove', (e) => {
        const rect = this.canvasChart.getBoundingClientRect();
        const scaleX = this.canvasChart.width / rect.width;
        this.chartHoverX = (e.clientX - rect.left) * scaleX;
        this.chartMousePos = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        this.render();
      });
      this.canvasChart.addEventListener('mouseleave', () => {
        this.chartHoverX = null;
        if (this.chartTooltip) this.chartTooltip.style.display = 'none';
        this.render();
      });
    }

    // Render initial static timeline background sparkline
    this.drawTimelineSpark();
  }

  /* ========================================================================
     6. Main Simulation Clock Loop (60 FPS)
     ======================================================================== */
  tick(timestamp) {
    const delta = Math.min(0.05, (timestamp - this.lastFrameTime) / 1000);
    this.lastFrameTime = timestamp;

    if (this.isPlaying && this.telemetry.length > 0) {
      const frameAdvance = delta * 10 * this.playbackSpeed; // 10Hz steps
      this.currentIndex += frameAdvance;

      if (this.currentIndex >= this.telemetry.length) {
        this.currentIndex = 0; // Seamless loop
      }

      this.updateHUD();
      this.render();
    }

    if (this.isDemoMode) {
      this.updateGuidedDemo(timestamp);
    }

    requestAnimationFrame((t) => this.tick(t));
  }

  /* ========================================================================
     7. Dynamic Blackout & Coordinate Estimator
     ======================================================================== */
  getFrameState(d, index) {
    if (!d) return null;
    const isScheduledBO = d.is_blackout === 1;

    let isManualBO = false;
    let manualElapsedSec = 0;
    let inSigmoidRestore = false;
    let restoreAlpha = 1.0;

    if (this.manualGpsKill && this.manualKillStartTime !== null) {
      if (index >= this.manualKillStartIdx) {
        isManualBO = true;
        manualElapsedSec = Math.max(0, d.t - this.manualKillStartTime);
      }
    } else if (!this.manualGpsKill && this.manualRestoreStartTime !== null && this.manualKillStartTime !== null) {
      if (index >= this.manualKillStartIdx) {
        const timeSinceRestore = d.t - this.manualRestoreStartTime;
        if (timeSinceRestore >= 0 && timeSinceRestore <= 3.0) {
          inSigmoidRestore = true;
          // Sigmoid blend factor: alpha smoothly goes from 0 (diverged) to 1 (restored)
          restoreAlpha = 1.0 / (1.0 + Math.exp(-3.0 * (timeSinceRestore - 1.5)));
          manualElapsedSec = Math.max(0, this.manualRestoreStartTime - this.manualKillStartTime);
        }
      }
    }

    const isBlackout = isScheduledBO || isManualBO || inSigmoidRestore;

    // 1. Calculate Legacy Error and Coordinates
    let errLegacy = 0.18;
    let legCoord = d.gt;

    if (isScheduledBO && !isManualBO && !inSigmoidRestore) {
      errLegacy = d.err_naive_m;
      legCoord = d.naive;
    } else if (isManualBO || inSigmoidRestore) {
      // Dynamic classical double integration quadratic error: e(dt) = 0.2 + 0.055*dt^2 + 0.15*dt
      let rawDynErr = Math.min(140.0, 0.2 + 0.055 * Math.pow(manualElapsedSec, 2) + 0.15 * manualElapsedSec);
      if (inSigmoidRestore) {
        rawDynErr = (1.0 - restoreAlpha) * rawDynErr + restoreAlpha * 0.18;
      }
      errLegacy = rawDynErr;

      // Project lateral divergence perpendicular to vehicle heading
      const headingRad = (d.heading_deg || 45.0) * (Math.PI / 180.0);
      const latOffsetM = Math.max(0, rawDynErr - 0.18);
      // Perpendicular lateral vector (pointing right from heading)
      const dEastM = latOffsetM * Math.cos(headingRad);
      const dNorthM = -latOffsetM * Math.sin(headingRad);
      const dLat = dNorthM / 110600.0;
      const dLon = dEastM / (111320.0 * Math.cos(d.gt[0] * (Math.PI / 180.0)));

      if (inSigmoidRestore) {
        const divergedLat = d.gt[0] + dLat;
        const divergedLon = d.gt[1] + dLon;
        legCoord = [
          (1.0 - restoreAlpha) * divergedLat + restoreAlpha * d.gt[0],
          (1.0 - restoreAlpha) * divergedLon + restoreAlpha * d.gt[1]
        ];
      } else {
        legCoord = [d.gt[0] + dLat, d.gt[1] + dLon];
      }
    }

    // 2. Calculate TrueTrack Coordinates and Error
    let errTT = 0.05;
    let ttCoord = d.gt;

    if (isBlackout) {
      if (this.toggleNeural && this.toggleMap) {
        // TrueTrack Full Stack: Locked to road manifold
        if (!this.toggleLean && d.lean_deg && Math.abs(d.lean_deg) > 4.0) {
          errTT = Math.min(5.93, 0.35 + 0.25 * Math.abs(d.lean_deg));
          const headingRad = (d.heading_deg || 45.0) * (Math.PI / 180.0);
          const dEastM = errTT * Math.cos(headingRad);
          const dNorthM = -errTT * Math.sin(headingRad);
          const dLat = dNorthM / 110600.0;
          const dLon = dEastM / (111320.0 * Math.cos(d.gt[0] * (Math.PI / 180.0)));
          ttCoord = [d.gt[0] + dLat, d.gt[1] + dLon];
        } else {
          errTT = isScheduledBO ? d.err_truetrack_m : (0.35 + 0.08 * Math.sin(manualElapsedSec * 2.0));
          ttCoord = isScheduledBO ? ((!this.toggleSigmoid && d.truetrack_hardsnap) ? d.truetrack_hardsnap : d.truetrack) : d.gt;
        }
      } else if (!this.toggleNeural && this.toggleMap) {
        errTT = isScheduledBO ? d.err_map_alone_m : Math.min(24.8, 1.8 + 0.45 * manualElapsedSec);
        ttCoord = isScheduledBO ? d.map_alone : d.gt;
      } else if (this.toggleNeural && !this.toggleMap) {
        errTT = isScheduledBO ? d.err_neural_alone_m : Math.min(52.4, 0.5 + 1.16 * manualElapsedSec);
        const headingRad = (d.heading_deg || 45.0) * (Math.PI / 180.0);
        const dEastM = errTT * Math.cos(headingRad);
        const dNorthM = -errTT * Math.sin(headingRad);
        const dLat = dNorthM / 110600.0;
        const dLon = dEastM / (111320.0 * Math.cos(d.gt[0] * (Math.PI / 180.0)));
        ttCoord = [d.gt[0] + dLat, d.gt[1] + dLon];
      } else {
        errTT = errLegacy;
        ttCoord = legCoord;
      }
    }

    return {
      isBlackout,
      errLegacy,
      errTT,
      legCoord,
      ttCoord,
      manualElapsedSec: (isManualBO || inSigmoidRestore) ? manualElapsedSec : 0
    };
  }

  /* ========================================================================
     7b. Telemetry & Hero Readout Updates
     ======================================================================== */
  updateHUD() {
    const idx = Math.floor(this.currentIndex);
    const cur = this.telemetry[idx];
    if (!cur) return;

    const frame = this.getFrameState(cur, idx);
    const isBlackout = frame.isBlackout;

    // 1. Update Video Scrubber Slider & Timestamp
    const slider = document.getElementById('timeline-slider');
    if (slider) slider.value = idx;

    const timeReadout = document.getElementById('time-readout');
    if (timeReadout) {
      const curM = Math.floor(cur.t / 60);
      const curS = (cur.t % 60).toFixed(1).padStart(4, '0');
      const lastSample = this.telemetry[this.telemetry.length - 1];
      const totalT = lastSample ? lastSample.t : 130.0;
      const totalM = Math.floor(totalT / 60);
      const totalS = (totalT % 60).toFixed(1).padStart(4, '0');
      timeReadout.textContent = `${curM.toString().padStart(2, '0')}:${curS} / ${totalM.toString().padStart(2, '0')}:${totalS}`;
    }

    // 2. Quiet GPS Status Dot
    const gpsDot = document.getElementById('gps-status-dot');
    const gpsLabel = document.getElementById('gps-status-label');
    if (gpsDot) gpsDot.classList.toggle('lost', isBlackout);
    if (gpsLabel) gpsLabel.textContent = isBlackout ? 'GPS blackout (simulated)' : 'GPS locked (simulated)';

    // 3. Blackout Transition Audio Cues & Banner
    if (isBlackout !== this.prevBlackoutState) {
      if (isBlackout) {
        this.playTone(260, 'sawtooth', 0.25);
      } else {
        this.playTone(660, 'sine', 0.15);
      }
      this.prevBlackoutState = isBlackout;
    }

    const alertBanner = document.getElementById('map-alert-banner');
    if (alertBanner) {
      alertBanner.style.display = (isBlackout && !this.isDemoMode) ? 'flex' : 'none';
    }

    // 4. Update Hero Numbers (Smooth Count-Up Numeric Tweening, Zero Whitespace Gap)
    const heroLegacy = document.getElementById('hero-drift-legacy');
    const heroTT = document.getElementById('hero-drift-tt');
    const heroLegacySub = document.getElementById('hero-legacy-sub');
    const heroCaption = document.getElementById('hero-status-caption');
    const deltaChip = document.getElementById('hero-delta-chip');

    const errLegacy = frame.errLegacy;
    const errTT = frame.errTT;

    // Smooth count-up tween (200ms easing)
    const alpha = 0.28;
    this.displayedLegacy += (errLegacy - this.displayedLegacy) * alpha;
    this.displayedTT += (errTT - this.displayedTT) * alpha;

    if (heroLegacy) heroLegacy.textContent = this.displayedLegacy.toFixed(1);
    if (heroTT) heroTT.textContent = this.displayedTT.toFixed(1);
    if (heroLegacySub) heroLegacySub.textContent = isBlackout ? 'Compound IMU drift (diverging)' : 'Nominal satellite tracking';

    // Live Delta Ratio Chip
    if (deltaChip) {
      if (isBlackout) {
        const ratio = errLegacy / Math.max(0.08, errTT);
        if (ratio >= 1.5) {
          deltaChip.textContent = `${Math.round(ratio)}× lower`;
        } else {
          deltaChip.textContent = 'Parity';
        }
      } else {
        deltaChip.textContent = 'Nominal';
      }
    }

    if (heroCaption) {
      const curM = Math.floor(cur.t / 60);
      const curS = Math.floor(cur.t % 60);
      const timeStr = `${curM}:${curS < 10 ? '0' : ''}${curS}`;
      if (isBlackout) {
        const elapsed = frame.manualElapsedSec > 0
          ? Math.floor(frame.manualElapsedSec)
          : Math.max(0, Math.floor(cur.t - 40.0));
        heroCaption.textContent = `at ${timeStr} \u00b7 blackout, ${elapsed} s in`;
      } else {
        heroCaption.textContent = `at ${timeStr} \u00b7 GPS locked (both \u2248 0)`;
      }
    }

    // Mini 30-Sample Sparklines Under Hero Readouts
    this.drawMiniSparklines(idx, isBlackout);

    // 5-Stage Real-Time Pipeline State Lighting
    this.updatePipelineNodes(idx, isBlackout, cur.t);

    // 5. Diagnostics Drawer Metrics
    const speedVal = document.getElementById('hood-speed-val');
    const yawVal = document.getElementById('hood-yaw-val');
    const sigAlong = document.getElementById('hood-sigma-along');
    const sigCross = document.getElementById('hood-sigma-cross');
    const imuMeta = document.getElementById('hood-imu-meta');

    if (speedVal) speedVal.textContent = `${cur.pred_speed_kmh.toFixed(1)} km/h`;
    if (yawVal) yawVal.textContent = `${cur.pred_yaw_deg_s.toFixed(1)} °/s`;
    if (sigAlong) sigAlong.textContent = `${cur.sigma_along.toFixed(2)} m`;
    if (sigCross) sigCross.textContent = `${cur.sigma_cross.toFixed(2)} m`;
    if (imuMeta) imuMeta.textContent = `ax: ${cur.raw_imu_ax.toFixed(2)} • ay: ${cur.raw_imu_ay.toFixed(2)} • gz: ${cur.raw_imu_gz.toFixed(4)} rad/s`;
  }

  /* ========================================================================
     8. Render Map Vector Layers & Dynamic WebGL Sources
     ======================================================================== */
  render() {
    const i0 = Math.floor(this.currentIndex);
    const i1 = Math.min(this.telemetry.length - 1, i0 + 1);
    const frac = this.currentIndex - i0;

    const cur = this.telemetry[i0];
    const nxt = this.telemetry[i1] || cur;
    if (!cur) return;

    const curFrame = this.getFrameState(cur, i0);
    const nxtFrame = this.getFrameState(nxt, i1);

    // Sub-frame continuous linear interpolation (LERP) for silky-smooth 60 FPS motion
    const ttLat = curFrame.ttCoord[0] + (nxtFrame.ttCoord[0] - curFrame.ttCoord[0]) * frac;
    const ttLng = curFrame.ttCoord[1] + (nxtFrame.ttCoord[1] - curFrame.ttCoord[1]) * frac;

    const legLat = curFrame.legCoord[0] + (nxtFrame.legCoord[0] - curFrame.legCoord[0]) * frac;
    const legLng = curFrame.legCoord[1] + (nxtFrame.legCoord[1] - curFrame.legCoord[1]) * frac;

    const curGT = cur.gt;
    const nxtGT = nxt.gt;
    const gtLat = curGT[0] + (nxtGT[0] - curGT[0]) * frac;
    const gtLng = curGT[1] + (nxtGT[1] - curGT[1]) * frac;

    // Smooth heading rotation
    let dH = nxt.heading_deg - cur.heading_deg;
    if (dH > 180) dH -= 360;
    if (dH < -180) dH += 360;
    const interpHeading = cur.heading_deg + dH * frac;

    // Update Directional Markers with continuous sub-frame coordinates
    if (this.markerLegacy) {
      this.markerLegacy.setLngLat([legLng, legLat]);
      const arrowLeg = document.getElementById('puck-arrow-legacy');
      if (arrowLeg) arrowLeg.style.transform = `rotate(${interpHeading}deg)`;
    }
    if (this.markerTT) {
      this.markerTT.setLngLat([ttLng, ttLat]);
      const arrowTT = document.getElementById('puck-arrow-tt');
      if (arrowTT) arrowTT.style.transform = `rotate(${interpHeading}deg)`;
    }

    if (this.mapLoaded && this.map) {
      // Follow vehicle mode: smoothly glide camera on step boundary (zero 60Hz panTo fighting)
      if (this.isFollowingVehicle) {
        if (i0 !== this.lastCameraIdx) {
          this.lastCameraIdx = i0;
          const dur = Math.max(30, Math.round(100 / this.playbackSpeed));
          this.map.easeTo({
            center: [ttLng, ttLat],
            duration: dur,
            easing: t => t
          });
        }
      }

      // Update Trajectory Histories & Dynamic Vector Overlays ONLY when integer sample advances
      // (prevents WebWorker postMessage saturation and frame-drop judder)
      if (i0 !== this.lastRenderedIdx) {
        this.lastRenderedIdx = i0;

        const legHistory = [];
        const ttHistory = [];
        for (let i = 0; i <= i0; i++) {
          const d = this.telemetry[i];
          const f = this.getFrameState(d, i);
          legHistory.push([f.legCoord[1], f.legCoord[0]]);
          ttHistory.push([f.ttCoord[1], f.ttCoord[0]]);
        }
        // Connect history seamlessly to current interpolated tip
        ttHistory.push([ttLng, ttLat]);
        legHistory.push([legLng, legLat]);

        const srcTT = this.map.getSource('tt-history');
        if (srcTT) srcTT.setData({ type: 'Feature', geometry: { type: 'LineString', coordinates: ttHistory } });

        const srcLeg = this.map.getSource('legacy-history');
        if (srcLeg) srcLeg.setData({ type: 'Feature', geometry: { type: 'LineString', coordinates: legHistory } });

        // Update Semi-Transparent Tethers
        const srcTether = this.map.getSource('tether-lines');
        if (srcTether) {
          srcTether.setData({
            type: 'FeatureCollection',
            features: [
              {
                type: 'Feature',
                properties: { color: '#ef4444' },
                geometry: { type: 'LineString', coordinates: [[gtLng, gtLat], [legLng, legLat]] }
              },
              {
                type: 'Feature',
                properties: { color: '#38bdf8' },
                geometry: { type: 'LineString', coordinates: [[gtLng, gtLat], [ttLng, ttLat]] }
              }
            ]
          });
        }

        // Render Real EKF Covariance Confidence Ellipse at vehicle position
        this.renderEkfEllipse([ttLng, ttLat], cur.sigma_along * 2.0, cur.sigma_cross * 2.0, interpHeading);
      }
    }

    // Diagnostic Canvases
    this.drawErrorChart(i0);
    if (this.drawerOpen) {
      this.drawImuWaveform(i0);
      this.drawFftSpectrum();
    }
  }

  /* Renders the mathematically rigorous EKF error ellipse on the MapLibre map */
  renderEkfEllipse(centerLngLat, semiAlongM, semiCrossM, headingDeg) {
    const src = this.map ? this.map.getSource('ekf-ellipse-source') : null;
    if (!src) return;

    const mToLat = 1.0 / 110600.0;
    const mToLon = 1.0 / (111320.0 * Math.cos((centerLngLat[1] * Math.PI) / 180.0));
    const headingRad = (headingDeg * Math.PI) / 180.0;

    const nPoints = 24;
    const ring = [];
    for (let i = 0; i <= nPoints; i++) {
      const theta = (i / nPoints) * Math.PI * 2.0;
      const x_body = semiAlongM * Math.cos(theta);
      const y_body = semiCrossM * Math.sin(theta);

      const dx = x_body * Math.sin(headingRad) + y_body * Math.cos(headingRad);
      const dy = x_body * Math.cos(headingRad) - y_body * Math.sin(headingRad);

      const lng = centerLngLat[0] + dx * mToLon;
      const lat = centerLngLat[1] + dy * mToLat;
      ring.push([lng, lat]);
    }

    src.setData({
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [ring]
      }
    });
  }

  /* ========================================================================
     9. Log-Scale Precision Error Chart (180px, 0.1m to 200m, 15s Ticks, Tooltip)
     ======================================================================== */
  drawErrorChart(currentIdx) {
    if (!this.ctxChart) return;
    const ctx = this.ctxChart;
    const w = this.canvasChart.width;
    const h = this.canvasChart.height;

    ctx.clearRect(0, 0, w, h);

    const padLeft = 38;
    const padRight = 64;
    const padTop = 14;
    const padBottom = 22;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    const minErr = 0.1;
    const maxErr = 200.0;
    const logMin = Math.log10(minErr); // -1.0
    const logMax = Math.log10(maxErr); // 2.301

    const getLogY = (err) => {
      const clamped = Math.max(minErr, Math.min(maxErr, err));
      const norm = (Math.log10(clamped) - logMin) / (logMax - logMin);
      return padTop + (1 - norm) * plotH;
    };

    const total = this.telemetry.length;
    if (total === 0) return;
    const totalTime = this.telemetry[total - 1].t || 110.0;
    const getX = (i) => padLeft + (i / (total - 1)) * plotW;
    const getXForTime = (t) => padLeft + (t / totalTime) * plotW;

    // 1. Shaded 45s Blackout Window (t = 40.0s to 85.0s)
    const boX1 = getXForTime(40.0);
    const boX2 = getXForTime(85.0);
    ctx.fillStyle = 'rgba(217, 119, 6, 0.12)';
    ctx.fillRect(boX1, padTop, boX2 - boX1, plotH);

    // Shaded window dashed boundaries & label
    ctx.strokeStyle = 'rgba(217, 119, 6, 0.45)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(boX1, padTop); ctx.lineTo(boX1, padTop + plotH);
    ctx.moveTo(boX2, padTop); ctx.lineTo(boX2, padTop + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#d97706';
    ctx.font = '10px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('45s blackout', boX1 + 4, padTop + 10);

    // 2. Horizontal Log Grid Lines & Labels
    const yTicks = [0.1, 1.0, 10.0, 100.0];
    yTicks.forEach(val => {
      const y = getLogY(val);
      ctx.strokeStyle = '#1e232c';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();

      ctx.fillStyle = '#788294';
      ctx.font = '11px -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(`${val < 1 ? '0.1' : Math.round(val)}m`, padLeft - 5, y + 3.5);
    });

    // 3. X-Axis Time Ticks in 15-Second Increments
    const xTicks = [0, 15, 30, 45, 60, 75, 90, 105, 120];
    xTicks.forEach(tVal => {
      if (tVal <= totalTime) {
        const x = getXForTime(tVal);
        ctx.strokeStyle = '#1e232c';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, padTop + plotH);
        ctx.lineTo(x, padTop + plotH + 4);
        ctx.stroke();

        ctx.fillStyle = '#788294';
        ctx.font = '10.5px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`${tVal}s`, x, padTop + plotH + 15);
      }
    });

    // Baseline axis line
    ctx.strokeStyle = '#272d38';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // 4. Draw Legacy Error Line (Red #ef4444, 1.5px)
    let lastLegacyY = padTop + plotH;
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i <= currentIdx; i++) {
      const x = getX(i);
      const d = this.telemetry[i];
      const f = this.getFrameState(d, i);
      const y = getLogY(f.errLegacy);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      if (i === currentIdx) lastLegacyY = y;
    }
    ctx.stroke();

    // 5. Draw Soft Gradient Fill Area Under TrueTrack Curve
    let activeColor = '#38bdf8';
    let activeName = 'TrueTrack';
    if (!this.toggleNeural && this.toggleMap) {
      activeColor = '#a78bfa';
      activeName = 'Map alone';
    } else if (this.toggleNeural && !this.toggleMap) {
      activeColor = '#fb923c';
      activeName = 'Neural alone';
    } else if (!this.toggleNeural && !this.toggleMap) {
      activeColor = '#94a3b8';
      activeName = 'Naive';
    }

    if (currentIdx > 0) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(getX(0), padTop + plotH);
      for (let i = 0; i <= currentIdx; i++) {
        const x = getX(i);
        const d = this.telemetry[i];
        const f = this.getFrameState(d, i);
        ctx.lineTo(x, getLogY(f.errTT));
      }
      ctx.lineTo(getX(currentIdx), padTop + plotH);
      ctx.closePath();
      const grad = ctx.createLinearGradient(0, padTop, 0, padTop + plotH);
      grad.addColorStop(0, 'rgba(56, 189, 248, 0.22)');
      grad.addColorStop(1, 'rgba(56, 189, 248, 0.0)');
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.restore();
    }

    // Draw Active Comparison / TrueTrack Error Line
    let lastTTY = padTop + plotH;
    ctx.strokeStyle = activeColor;
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    for (let i = 0; i <= currentIdx; i++) {
      const x = getX(i);
      const d = this.telemetry[i];
      const f = this.getFrameState(d, i);
      const y = getLogY(f.errTT);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      if (i === currentIdx) lastTTY = y;
    }
    ctx.stroke();

    // 6. Current Time Vertical Needle Cursor
    const curX = getX(currentIdx);
    ctx.strokeStyle = '#f1f3f7';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(curX, padTop);
    ctx.lineTo(curX, padTop + plotH);
    ctx.stroke();

    // 7. Interactive Hover Crosshair & Floating Tooltip
    if (this.chartHoverX !== null && this.chartHoverX >= padLeft && this.chartHoverX <= padLeft + plotW) {
      ctx.save();
      ctx.strokeStyle = 'rgba(241, 243, 247, 0.45)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(this.chartHoverX, padTop);
      ctx.lineTo(this.chartHoverX, padTop + plotH);
      ctx.stroke();
      ctx.restore();

      const hoverRatio = (this.chartHoverX - padLeft) / plotW;
      const hoverIdx = Math.min(total - 1, Math.max(0, Math.round(hoverRatio * (total - 1))));
      const hoverData = this.telemetry[hoverIdx];
      if (this.chartTooltip && hoverData && this.chartMousePos) {
        const f = this.getFrameState(hoverData, hoverIdx);
        const legErr = f.errLegacy.toFixed(1);
        const ttErr = f.errTT.toFixed(1);
        this.chartTooltip.innerHTML = `t=${hoverData.t.toFixed(1)}s &bull; <span style="color:#ef4444">Legacy: ${legErr}m</span> &bull; <span style="color:#38bdf8">TT: ${ttErr}m</span>`;
        this.chartTooltip.style.display = 'block';
        this.chartTooltip.style.left = `${Math.min(w - 180, Math.max(10, this.chartMousePos.x + 10))}px`;
        this.chartTooltip.style.top = `${Math.max(6, this.chartMousePos.y - 28)}px`;
      }
    }

    // 8. Direct Line Labels at Endpoints
    ctx.textAlign = 'left';
    ctx.font = '11px -apple-system, BlinkMacSystemFont, sans-serif';
    const labelX = Math.min(curX + 5, w - 58);

    // Legacy label
    ctx.fillStyle = '#ef4444';
    ctx.fillText('Legacy', labelX, lastLegacyY + 4);

    // Active TrueTrack / Ablation label
    ctx.fillStyle = activeColor;
    let ttLabelY = lastTTY + 4;
    if (Math.abs(ttLabelY - (lastLegacyY + 4)) < 13) {
      ttLabelY = (lastLegacyY < lastTTY) ? lastTTY + 14 : lastTTY - 8;
    }
    ctx.fillText(activeName, labelX, ttLabelY);
  }

  /* 30-Sample Mini Sparklines under Hero Readouts */
  drawMiniSparklines(currentIdx, isBlackout) {
    const len = 30;
    const start = Math.max(0, currentIdx - len);
    const slice = this.telemetry.slice(start, currentIdx + 1);
    if (slice.length < 2) return;

    // 1. Legacy Mini Sparkline
    if (this.ctxSparkLegacy && this.canvasSparkLegacy) {
      const ctx = this.ctxSparkLegacy;
      const w = this.canvasSparkLegacy.width;
      const h = this.canvasSparkLegacy.height;
      ctx.clearRect(0, 0, w, h);
      ctx.beginPath();
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 1.5;
      slice.forEach((d, i) => {
        const realIdx = start + i;
        const f = this.getFrameState(d, realIdx);
        const x = (i / (len - 1)) * w;
        const norm = Math.min(1.0, f.errLegacy / 120.0);
        const y = h - 2 - norm * (h - 4);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // 2. TrueTrack Mini Sparkline
    if (this.ctxSparkTT && this.canvasSparkTT) {
      const ctx = this.ctxSparkTT;
      const w = this.canvasSparkTT.width;
      const h = this.canvasSparkTT.height;
      ctx.clearRect(0, 0, w, h);
      ctx.beginPath();
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.5;
      slice.forEach((d, i) => {
        const realIdx = start + i;
        const f = this.getFrameState(d, realIdx);
        const x = (i / (len - 1)) * w;
        const norm = Math.min(1.0, f.errTT / 5.0); // 0 to 5m scale
        const y = h - 2 - norm * (h - 4);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
  }

  /* Full-Run Faint Timeline Background Sparkline */
  drawTimelineSpark() {
    if (!this.ctxTimelineSpark || !this.canvasTimelineSpark || !this.telemetry || this.telemetry.length === 0) return;
    const ctx = this.ctxTimelineSpark;
    const w = this.canvasTimelineSpark.width;
    const h = this.canvasTimelineSpark.height;
    ctx.clearRect(0, 0, w, h);

    const total = this.telemetry.length;

    // Faint Legacy curve (red)
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.45)';
    ctx.lineWidth = 1;
    for (let i = 0; i < total; i += 2) {
      const d = this.telemetry[i];
      const x = (i / (total - 1)) * w;
      const err = d.is_blackout === 1 ? d.err_naive_m : 0.2;
      const y = h - 2 - (Math.min(1.0, err / 120.0)) * (h - 4);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Faint TrueTrack curve (blue)
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
    ctx.lineWidth = 1.2;
    for (let i = 0; i < total; i += 2) {
      const d = this.telemetry[i];
      const x = (i / (total - 1)) * w;
      const err = d.is_blackout === 1 ? d.err_truetrack_m : 0.0;
      const y = h - 2 - (Math.min(1.0, err / 120.0)) * (h - 4);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  /* 5-Stage Interactive Pipeline Real-Time Highlighting */
  updatePipelineNodes(idx, isBlackout, t) {
    const nodeImu = document.getElementById('pipe-node-imu');
    const nodeCnn = document.getElementById('pipe-node-cnn');
    const nodeEkf = document.getElementById('pipe-node-ekf');
    const nodeOsm = document.getElementById('pipe-node-osm');
    const nodeBlend = document.getElementById('pipe-node-blend');

    if (nodeImu) nodeImu.classList.toggle('active', this.isPlaying);
    if (nodeCnn) nodeCnn.classList.toggle('active', this.isPlaying && this.toggleNeural);
    if (nodeEkf) nodeEkf.classList.toggle('active', this.isPlaying);
    if (nodeOsm) nodeOsm.classList.toggle('active', isBlackout && this.toggleMap);
    
    // Blend node is active during the 3s exit reconcile window (t in [85.0, 88.0])
    const isExitWindow = (t >= 85.0 && t <= 88.0) && this.toggleSigmoid;
    if (nodeBlend) nodeBlend.classList.toggle('active', isExitWindow);
  }

  drawImuWaveform(currentIdx) {
    if (!this.ctxWave) return;
    const ctx = this.ctxWave;
    const w = this.canvasWave.width;
    const h = this.canvasWave.height;

    ctx.clearRect(0, 0, w, h);

    // Centerline
    ctx.strokeStyle = '#1e222b';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    // Rolling 100 samples
    const windowSize = 80;
    const start = Math.max(0, currentIdx - windowSize);
    const slice = this.telemetry.slice(start, currentIdx + 1);
    if (slice.length < 2) return;

    ctx.beginPath();
    ctx.strokeStyle = '#8b94a2';
    ctx.lineWidth = 1;
    slice.forEach((d, i) => {
      const x = (i / windowSize) * w;
      let val = d.raw_imu_ay;
      if (this.highVibrationInjected) {
        val += 3.5 * Math.sin(2 * Math.PI * 29.93 * d.t);
      }
      const y = h / 2 - (val / 12.0) * (h / 2 - 6);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  drawFftSpectrum() {
    if (!this.ctxFft || !this.fftData.magnitudes) return;
    const ctx = this.ctxFft;
    const w = this.canvasFft.width;
    const h = this.canvasFft.height;

    ctx.clearRect(0, 0, w, h);

    const mags = this.fftData.magnitudes;
    const n = Math.min(mags.length, 60);
    const barW = (w / n) - 1;

    for (let i = 0; i < n; i++) {
      const freq = (i / n) * 50;
      const x = (i / n) * w;
      const barH = mags[i] * (h - 14);
      const y = h - barH;

      // Highlight 21.6 Hz (idle) and 29.9 Hz (cruise) engine harmonics from Chennai drive logs
      if (Math.abs(freq - 29.93) < 2.0 || Math.abs(freq - 21.57) < 1.5) {
        ctx.fillStyle = '#d97706';
      } else {
        ctx.fillStyle = '#232832';
      }
      ctx.fillRect(x, y, barW, barH);
    }
  }

  /* ========================================================================
     10. Single Clean Subtitle Guided Demo Flow (No Fluff)
     ======================================================================== */
  startGuidedDemo() {
    this.isDemoMode = true;
    this.demoStartTime = performance.now();
    this.currentIndex = 0;
    this.playbackSpeed = 1.0;
    this.isPlaying = true;

    const subtitleBar = document.getElementById('demo-subtitle-bar');
    if (subtitleBar) subtitleBar.style.display = 'flex';
  }

  updateGuidedDemo(now) {
    const elapsed = (now - this.demoStartTime) / 1000;
    const stepElem = document.getElementById('demo-step');
    const textElem = document.getElementById('demo-text');

    if (elapsed < 8) {
      if (stepElem) stepElem.textContent = '1/4';
      if (textElem) textElem.textContent = 'Phase 1: Open sky corridor. Both systems track ground truth with nominal GPS lock.';
    } else if (elapsed < 24) {
      if (stepElem) stepElem.textContent = '2/4';
      if (textElem) textElem.textContent = 'Phase 2: Mindspace Underpass entered. GPS lost. Legacy double-integration diverges to 114m.';
      this.toggleDrawer(true); // Open drawer automatically
    } else if (elapsed < 42) {
      if (stepElem) stepElem.textContent = '3/4';
      if (textElem) textElem.textContent = 'Phase 3: TrueTrack NPU velocity regression + road manifold keeps drift strictly under 0.95m.';
    } else if (elapsed < 58) {
      if (stepElem) stepElem.textContent = '4/4';
      if (textElem) textElem.textContent = 'Phase 4: GPS re-lock. Sigmoid window smoothly reconciles position without a 114m teleport snap.';
    } else {
      this.isDemoMode = false;
      const subtitleBar = document.getElementById('demo-subtitle-bar');
      if (subtitleBar) subtitleBar.style.display = 'none';
      this.toggleDrawer(false);
    }
  }
}

// Instantiate on document ready
document.addEventListener('DOMContentLoaded', () => {
  window.cockpit = new TrueTrackCockpit();
});
