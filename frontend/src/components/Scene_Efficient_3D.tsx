// components/Scene.tsx
import React, { useEffect, useRef, useCallback, useState, useMemo } from 'react';
import { Canvas, useThree, ThreeEvent } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

export interface SimulationData {
  nx: number;
  ny: number;
  dx: number;
  dy: number;
  pressures: number[][]; // shape: [timeStep][nx * ny]
  timeSteps: number[];
}

export interface HoverInfo {
  i: number;
  j: number;
  pressure: number;
}

export interface SceneProps {
  simulationData: SimulationData;
  currentTimeIndex?: number;
  gridOpacity?: number;
  onHover?: (hoverInfo: HoverInfo | null) => void;
  cameraDistance?: number;
  is3D?: boolean;
  heightExaggeration?: number;
  restrictVertical?: boolean;
}

// Constants
const VISUAL_SCALE = 0.2; // scales the grid dimensions

// Maps a pressure value to a color using a blue-to-red scale.
function pressureToColor(p: number, minP: number, maxP: number): THREE.Color {
  const fraction = (p - minP) / Math.max(1e-30, maxP - minP);
  const hue = 240 - 240 * fraction; // 240 = blue, 0 = red
  const color = new THREE.Color();
  color.setHSL(hue / 360, 1, 0.55);
  return color;
}

/* ------------------------------------------------------------------
   1) ReservoirBlocksInstanced: using InstancedMesh with bottom-anchored geometry
------------------------------------------------------------------- */
function ReservoirBlocksInstanced({
  simulationData,
  currentTimeIndex = 0,
  gridOpacity = 1,
  onHover,
  is3D = true,
  heightExaggeration = 1,
}: SceneProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { nx, ny, dx, dy, pressures } = simulationData;
  const instanceCount = nx * ny;
  const cellWidth = dx * VISUAL_SCALE;
  const cellHeight = dy * VISUAL_SCALE;

  // Create a box geometry exactly sized for each cell.
  // Shift the geometry upward by 0.5 so the bottom is at z=0.
  const boxGeom = useMemo(() => {
    const geom = new THREE.BoxGeometry(cellWidth, cellHeight, 1);
    geom.translate(0, 0, 0.5);
    return geom;
  }, [cellWidth, cellHeight]);

  // Precompute a base translation matrix for each cell (XY only)
  const [instanceInfo] = useState(() => {
    const arr: { matrix: THREE.Matrix4; i: number; j: number }[] = [];
    const totalWidth = nx * cellWidth;
    const totalHeight = ny * cellHeight;
    const halfW = totalWidth / 2;
    const halfH = totalHeight / 2;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const xPos = i * cellWidth - halfW + cellWidth / 2;
        const yPos = j * cellHeight - halfH + cellHeight / 2;
        const m = new THREE.Matrix4().makeTranslation(xPos, yPos, 0);
        arr.push({ matrix: m, i, j });
      }
    }
    return arr;
  });

  // Instanced buffer attribute for per-instance colors.
  const [colorAttribute] = useState(() => {
    const arr = new Float32Array(instanceCount * 3);
    return new THREE.InstancedBufferAttribute(arr, 3);
  });

  // Create the material only once.
  const material = useMemo(() => {
    const mat = new THREE.MeshStandardMaterial({
      metalness: 0,
      roughness: 0.6,
      opacity: gridOpacity,
      transparent: gridOpacity < 1,
      side: THREE.DoubleSide,
    });
    mat.vertexColors = false; // We'll update instanceColor separately.
    return mat;
  }, []); // Do not include gridOpacity so that it doesn't recreate the material.

  // Smoothly update material opacity when gridOpacity changes.
  useEffect(() => {
    material.opacity = gridOpacity;
    material.transparent = gridOpacity < 1;
    material.needsUpdate = true;
  }, [gridOpacity, material]);

  // Update instance matrices and colors when currentTimeIndex changes.
  useEffect(() => {
    if (!meshRef.current || !pressures || !pressures[currentTimeIndex]) return;
    const instancedMesh = meshRef.current;
    const currentPressures = pressures[currentTimeIndex];
    const finalPressures = pressures[pressures.length - 1];
    const currentMin = Math.min(...currentPressures);
    const currentMax = Math.max(...currentPressures);
    const finalMin = Math.min(...finalPressures);
    const finalMax = Math.max(...finalPressures);
    // Compute pressure difference in MPa for height scaling.
    const diffMPa = (finalMax - finalMin) / 1e6;

    for (let idx = 0; idx < instanceCount; idx++) {
      const { matrix, i, j } = instanceInfo[idx];
      const pressureVal = currentPressures[i + j * nx];
      // Update color based on current timestep.
      const color = pressureToColor(pressureVal, currentMin, currentMax);
      colorAttribute.setXYZ(idx, color.r, color.g, color.b);

      // Height: minimum pressure maps to 0, maximum maps to diffMPa * heightExaggeration.
      const fractionHeight = (pressureVal - finalMin) / Math.max(1e-30, finalMax - finalMin);
      const zThickness = is3D ? fractionHeight * diffMPa * heightExaggeration : 0.01;
      // With the geometry pivot shifted to the bottom, scaling will affect only the top.
      const scaleM = new THREE.Matrix4().makeScale(1, 1, zThickness);
      const finalMatrix = new THREE.Matrix4().copy(matrix).multiply(scaleM);
      instancedMesh.setMatrixAt(idx, finalMatrix);
    }
    colorAttribute.needsUpdate = true;
    instancedMesh.instanceMatrix.needsUpdate = true;
  }, [currentTimeIndex, pressures, instanceCount, instanceInfo, nx, colorAttribute, is3D, heightExaggeration]);

  const handlePointerMove = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      if (!onHover) return;
      e.stopPropagation();
      const instId = e.instanceId;
      if (instId == null) {
        onHover(null);
        return;
      }
      const { i, j } = instanceInfo[instId];
      const pVal = pressures[currentTimeIndex][i + j * nx];
      onHover({ i, j, pressure: pVal });
    },
    [onHover, instanceInfo, pressures, nx, currentTimeIndex]
  );

  const handlePointerOut = useCallback(() => {
    onHover && onHover(null);
  }, [onHover]);

  return (
    <instancedMesh
      ref={meshRef}
      args={[boxGeom, material, instanceCount]}
      instanceColor={colorAttribute}
      castShadow
      receiveShadow
      onPointerMove={handlePointerMove}
      onPointerOut={handlePointerOut}
    />
  );
}

/* ------------------------------------------------------------------
   2) SceneContents: Sets up camera, minimal lights, and a black background.
------------------------------------------------------------------- */
const SceneContents: React.FC<SceneProps> = (props) => {
  const { simulationData, cameraDistance = 100 } = props;
  const { camera, gl, scene } = useThree();

  // Update camera position relative to its current direction.
  const prevCameraDistance = useRef(cameraDistance);
  useEffect(() => {
    const diff = cameraDistance - prevCameraDistance.current;
    const direction = new THREE.Vector3();
    camera.getWorldDirection(direction);
    camera.position.add(direction.multiplyScalar(diff));
    prevCameraDistance.current = cameraDistance;
    camera.updateProjectionMatrix();
  }, [cameraDistance, camera]);

  useEffect(() => {
    gl.shadowMap.enabled = true;
    gl.shadowMap.type = THREE.PCFSoftShadowMap;
    scene.background = new THREE.Color('#000000');
  }, [gl, scene]);

  return (
    <>
      <OrbitControls enableZoom={false} />
      <ambientLight intensity={0.5} />
      <directionalLight position={[20, 30, 50]} intensity={0.4} castShadow />
      {simulationData && <ReservoirBlocksInstanced {...props} />}
    </>
  );
};

/* ------------------------------------------------------------------
   3) Top-level Scene component with Canvas.
   The Canvas is mounted once so that control changes do not remount it.
------------------------------------------------------------------- */
const Scene: React.FC<SceneProps> = (props) => {
  const { simulationData, cameraDistance = 100 } = props;
  if (!simulationData) return null;
  return (
    <Canvas
      style={{ width: '100%', height: '100%' }}
      camera={{ position: [0, 0, cameraDistance], fov: 45, near: 0.5, far: 5000 }}
      shadows
    >
      <SceneContents {...props} />
    </Canvas>
  );
};

export default Scene;
