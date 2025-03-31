// pages/index.tsx
import { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import type { NextPage } from 'next';
import styles from '../styles/Home.module.css';
import { SimulationData, HoverInfo } from '../components/Scene';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import * as THREE from 'three';

const Scene = dynamic(() => import('../components/Scene'), { ssr: false });

const Home: NextPage = () => {
  const [simulationData, setSimulationData] = useState<SimulationData | null>(null);
  const [currentTimeIndex, setCurrentTimeIndex] = useState<number>(0);
  const [gridOpacity, setGridOpacity] = useState<number>(1);
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [cameraDistance, setCameraDistance] = useState<number>(140);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [is3D, setIs3D] = useState<boolean>(true);
  const [heightExaggeration, setHeightExaggeration] = useState<number>(1);
  const [restrictVertical, setRestrictVertical] = useState<boolean>(false);

  // Clipping plane controls.
  const [showClippingPlane, setShowClippingPlane] = useState<boolean>(false);
  const [clippingPlanePosition, setClippingPlanePosition] = useState<{ x: number; y: number; z: number }>({
    x: 0,
    y: 0,
    z: 0,
  });
  const [clippingPlaneRotation, setClippingPlaneRotation] = useState<{ x: number; y: number; z: number }>({
    x: 0,
    y: 0,
    z: 0,
  });

  // Since we no longer render a helper mesh for the clipping plane, external rotation buttons update state only.
  const visualScale = 0.2;
  const computedMaxZoom = simulationData
    ? 4 * simulationData.nx * simulationData.dx * visualScale
    : 140;

  const handleLoadData = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:5000/simulation_data');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: SimulationData = await response.json();
      if (!data.pressures || data.pressures.length === 0) {
        throw new Error('No simulation data loaded.');
      }
      setSimulationData(data);
      setCurrentTimeIndex(0);
      setCameraDistance(2 * data.nx * data.dx * visualScale);
    } catch (err) {
      console.error('Error loading data:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRerun = async () => {
    setIsLoading(true);
    try {
      const runResponse = await fetch('http://localhost:5000/run', { method: 'POST' });
      if (!runResponse.ok) {
        throw new Error(`Run endpoint failed. status: ${runResponse.status}`);
      }
      await handleLoadData();
    } catch (err) {
      console.error('Error re-running simulation:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Playback animation logic.
  const animRef = useRef<number>();
  const frameFractionRef = useRef<number>(0);
  const lastFrameTimeRef = useRef<number>(0);

  useEffect(() => {
    if (!simulationData || !isPlaying) {
      frameFractionRef.current = currentTimeIndex;
      return;
    }
    const totalSteps = simulationData.timeSteps.length;
    const baseDuration = 10000;
    const duration = baseDuration / playbackSpeed;
    frameFractionRef.current = currentTimeIndex;
    const animate = (time: number) => {
      if (!lastFrameTimeRef.current) {
        lastFrameTimeRef.current = time;
      }
      const dt = time - lastFrameTimeRef.current;
      lastFrameTimeRef.current = time;
      const stepInc = (dt / duration) * totalSteps;
      frameFractionRef.current += stepInc;
      if (frameFractionRef.current >= totalSteps) {
        frameFractionRef.current %= totalSteps;
      }
      const newIndex = Math.floor(frameFractionRef.current);
      setCurrentTimeIndex(newIndex);
      animRef.current = requestAnimationFrame(animate);
    };
    animRef.current = requestAnimationFrame(animate);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      lastFrameTimeRef.current = 0;
    };
  }, [isPlaying, simulationData, playbackSpeed]);

  // External rotation buttons update clippingPlaneRotation state.
  const rotateClippingPlane = (axis: 'x' | 'y' | 'z') => {
    setClippingPlaneRotation((prev) => ({ ...prev, [axis]: prev[axis] + 90 }));
  };

  return (
    <div className={styles.appContainer}>
      <header className={styles.header}>
        <h1>Reservoir Simulator</h1>
      </header>
      <main className={styles.mainView}>
        {simulationData ? (
          <Scene
            simulationData={simulationData}
            currentTimeIndex={currentTimeIndex}
            gridOpacity={gridOpacity}
            onHover={setHoverInfo}
            cameraDistance={cameraDistance}
            is3D={is3D}
            heightExaggeration={heightExaggeration}
            restrictVertical={restrictVertical}
            showClippingPlane={showClippingPlane}
            clippingPlanePosition={clippingPlanePosition}
            clippingPlaneRotation={clippingPlaneRotation}
          />
        ) : (
          <div className={styles.noDataMessage}>
            <p>No simulation data loaded.</p>
          </div>
        )}
      </main>
      <aside className={styles.rightPanel}>
        <h2>Controls</h2>
        <section className={styles.dataControls}>
          <h3>Data Controls</h3>
          <div className={styles.buttonGroup}>
            <button onClick={handleLoadData} disabled={isLoading}>
              {isLoading ? 'Loading Data...' : 'Load Data'}
            </button>
            <button onClick={handleRerun} disabled={isLoading || !simulationData} style={{ marginLeft: '10px' }}>
              {isLoading ? 'Running Simulation...' : 'Re-Run Simulation'}
            </button>
          </div>
        </section>
        <section className={styles.playbackControls}>
          <h3>Simulation Playback</h3>
          <div className={styles.playbackRow}>
            <div className={styles.sliderGroup}>
              <label htmlFor="timeStep">Time Step:</label>
              <input
                id="timeStep"
                type="range"
                min="0"
                max={simulationData ? simulationData.timeSteps.length - 1 : 0}
                value={currentTimeIndex}
                onChange={(e) => setCurrentTimeIndex(Number(e.target.value))}
              />
              <div>Current Time: {simulationData?.timeSteps[currentTimeIndex]} s</div>
            </div>
            <button className={styles.playButton} onClick={() => setIsPlaying(!isPlaying)}>
              {isPlaying ? <PauseIcon fontSize="small" /> : <PlayArrowIcon fontSize="small" />}
            </button>
          </div>
          <div className={styles.sliderGroup}>
            <label htmlFor="playbackSpeed">Playback Speed:</label>
            <input
              id="playbackSpeed"
              type="range"
              min="0.5"
              max="3"
              step="0.1"
              value={playbackSpeed}
              onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
            />
            <div>Speed: {playbackSpeed}x</div>
          </div>
        </section>
        <section className={styles.visualizationControls}>
          <h3>Visualization Mode</h3>
          <button onClick={() => setIs3D(!is3D)}>{is3D ? 'Switch to 2D Mode' : 'Switch to 3D Mode'}</button>
        </section>
        <section className={styles.visualizationControls}>
          <h3>Visualization Controls</h3>
          <div className={styles.sliderGroup}>
            <label htmlFor="gridOpacity">Grid Opacity:</label>
            <input
              id="gridOpacity"
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={gridOpacity}
              onChange={(e) => setGridOpacity(Number(e.target.value))}
            />
            <div>Opacity: {gridOpacity}</div>
          </div>
          <div className={styles.sliderGroup}>
            <label htmlFor="zoom">Zoom:</label>
            <input
              id="zoom"
              type="range"
              min="5"
              max={computedMaxZoom}
              step="0.5"
              value={cameraDistance}
              onChange={(e) => setCameraDistance(Number(e.target.value))}
            />
            <div>Camera Distance: {cameraDistance}</div>
          </div>
          <div className={styles.sliderGroup}>
            <label htmlFor="heightExaggeration">Height Exaggeration:</label>
            <input
              id="heightExaggeration"
              type="range"
              min="0.5"
              max="30"
              step="0.1"
              value={heightExaggeration}
              onChange={(e) => setHeightExaggeration(Number(e.target.value))}
            />
            <div>Exaggeration: {heightExaggeration}x</div>
          </div>
          <div className={styles.sliderGroup}>
            <label htmlFor="verticalRotation">Restrict Vertical Rotation:</label>
            <button onClick={() => setRestrictVertical(!restrictVertical)}>
              {restrictVertical ? 'Horizontal Only' : 'Free Rotation'}
            </button>
          </div>
        </section>
        <section className={styles.visualizationControls}>
          <h3>Cross-Section (X-ray) Mode</h3>
          <div className={styles.sliderGroup}>
            <label htmlFor="clipEnabled">Enable Clipping:</label>
            <input
              id="clipEnabled"
              type="checkbox"
              checked={showClippingPlane}
              onChange={(e) => setShowClippingPlane(e.target.checked)}
            />
          </div>
          {showClippingPlane && (
            <div className={styles.sliderGroup}>
              <div style={{ marginBottom: '5px' }}>Clipping Plane Position:</div>
              <div>
                <label>Pos X:</label>
                <input
                  type="range"
                  min="-100"
                  max="100"
                  step="1"
                  value={clippingPlanePosition.x}
                  onChange={(e) =>
                    setClippingPlanePosition((prev) => ({ ...prev, x: Number(e.target.value) }))
                  }
                />
              </div>
              <div>
                <label>Pos Y:</label>
                <input
                  type="range"
                  min="-100"
                  max="100"
                  step="1"
                  value={clippingPlanePosition.y}
                  onChange={(e) =>
                    setClippingPlanePosition((prev) => ({ ...prev, y: Number(e.target.value) }))
                  }
                />
              </div>
              <div>
                <label>Pos Z:</label>
                <input
                  type="range"
                  min="-100"
                  max="100"
                  step="1"
                  value={clippingPlanePosition.z}
                  onChange={(e) =>
                    setClippingPlanePosition((prev) => ({ ...prev, z: Number(e.target.value) }))
                  }
                />
              </div>
              <div style={{ marginTop: '5px' }}>Clipping Plane Rotation (deg):</div>
              <div>
                <label>Rot X:</label>
                <input
                  type="range"
                  min="0"
                  max="360"
                  step="1"
                  value={clippingPlaneRotation.x}
                  onChange={(e) =>
                    setClippingPlaneRotation((prev) => ({ ...prev, x: Number(e.target.value) }))
                  }
                />
              </div>
              <div>
                <label>Rot Y:</label>
                <input
                  type="range"
                  min="0"
                  max="360"
                  step="1"
                  value={clippingPlaneRotation.y}
                  onChange={(e) =>
                    setClippingPlaneRotation((prev) => ({ ...prev, y: Number(e.target.value) }))
                  }
                />
              </div>
              <div>
                <label>Rot Z:</label>
                <input
                  type="range"
                  min="0"
                  max="360"
                  step="1"
                  value={clippingPlaneRotation.z}
                  onChange={(e) =>
                    setClippingPlaneRotation((prev) => ({ ...prev, z: Number(e.target.value) }))
                  }
                />
              </div>
              <div style={{ marginTop: '5px' }}>
                <button onClick={() => rotateClippingPlane('x', clippingPlaneRef)} style={{ marginRight: '5px' }}>
                  Rotate X +90°
                </button>
                <button onClick={() => rotateClippingPlane('y', clippingPlaneRef)} style={{ marginRight: '5px' }}>
                  Rotate Y +90°
                </button>
                <button onClick={() => rotateClippingPlane('z', clippingPlaneRef)}>Rotate Z +90°</button>
              </div>
            </div>
          )}
        </section>
        <section className={styles.infoSection}>
          <h3>Hovered Cell Info</h3>
          {hoverInfo ? (
            <div>
              Pressure: {(hoverInfo.pressure / 1e6).toFixed(2)} MPa
              <br />
              Coordinates: [{hoverInfo.i}, {hoverInfo.j}]
            </div>
          ) : (
            <div>Hover over a grid cell</div>
          )}
        </section>
        <section className={styles.legendSection}>
          <h3>Legend</h3>
          <div className={styles.legendContainer}>
            <div className={styles.legendGradient}></div>
            <div className={styles.legendLabels}>
              <span>High Pressure (MPa)</span>
              <span>Low Pressure (MPa)</span>
            </div>
          </div>
        </section>
      </aside>
    </div>
  );
};

export default Home;
