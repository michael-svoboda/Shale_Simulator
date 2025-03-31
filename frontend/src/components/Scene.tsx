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
  pressures: number[][]; // [timeStep][nx * ny]
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
  // When true, apply clipping.
  showClippingPlane?: boolean;
  // External controls for the clipping plane (position and rotation in degrees)
  clippingPlanePosition?: { x: number; y: number; z: number };
  clippingPlaneRotation?: { x: number; y: number; z: number };
}

const VISUAL_SCALE = 0.2;

function pressureToColor(p: number, minP: number, maxP: number): THREE.Color {
  const fraction = (p - minP) / Math.max(1e-30, maxP - minP);
  const hue = 240 - 240 * fraction; // 240 = blue, 0 = red
  const color = new THREE.Color();
  color.setHSL(hue / 360, 1, 0.55);
  return color;
}

/* ------------------------------------------------------------------
   ReservoirBlocksInstanced:
   Renders the reservoir cells via InstancedMesh.
   If a clippingPlane is provided, its values are passed into the shader.
------------------------------------------------------------------- */
interface BlockProps extends SceneProps {
  clippingPlane?: THREE.Plane;
}
function ReservoirBlocksInstanced({
  simulationData,
  currentTimeIndex = 0,
  gridOpacity = 1,
  onHover,
  is3D = true,
  heightExaggeration = 1,
  clippingPlane,
}: BlockProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { nx, ny, dx, dy, pressures } = simulationData;
  const instanceCount = nx * ny;
  const cellWidth = dx * VISUAL_SCALE;
  const cellHeight = dy * VISUAL_SCALE;

  const boxGeom = useMemo(() => {
    const geom = new THREE.BoxGeometry(cellWidth, cellHeight, 1);
    geom.translate(0, 0, 0.5);
    return geom;
  }, [cellWidth, cellHeight]);

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

  const [colorAttribute] = useState(() => {
    const arr = new Float32Array(instanceCount * 3);
    return new THREE.InstancedBufferAttribute(arr, 3);
  });

  const material = useMemo(() => {
    const mat = new THREE.MeshStandardMaterial({
      metalness: 0,
      roughness: 0.6,
      opacity: gridOpacity,
      transparent: gridOpacity < 1,
      side: THREE.DoubleSide,
      clipShadows: true,
    });
    mat.vertexColors = false;
    mat.onBeforeCompile = (shader) => {
      shader.uniforms.uClippingPlane = { value: new THREE.Vector4(0, 0, 0, 0) };
      // Pass world position to fragment shader.
      shader.vertexShader = 'varying vec3 vWorldPosition;\n' + shader.vertexShader;
      shader.vertexShader = shader.vertexShader.replace(
        '#include <worldpos_vertex>',
        `#include <worldpos_vertex>
         vWorldPosition = worldPosition.xyz;`
      );
      // Instead of discarding, we smoothly blend a cap color.
      shader.fragmentShader =
        'uniform vec4 uClippingPlane;\nvarying vec3 vWorldPosition;\n' + shader.fragmentShader;
      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <clipping_planes_fragment>',
        `#include <clipping_planes_fragment>
         float capWidth = 0.05;
         float d = dot(uClippingPlane.xyz, vWorldPosition) + uClippingPlane.w;
         if(d > capWidth) discard;
         else if(d > 0.0) {
           float factor = smoothstep(0.0, capWidth, d);
           gl_FragColor = mix(gl_FragColor, vec4(0.0, 0.0, 0.0, 0.0), factor);
         }`
      );
      mat.userData.shader = shader;
    };
    return mat;
  }, []);

  useEffect(() => {
    material.opacity = gridOpacity;
    material.transparent = gridOpacity < 1;
    material.needsUpdate = true;
  }, [gridOpacity, material]);

  useEffect(() => {
    if (material.userData.shader) {
      if (clippingPlane) {
        material.userData.shader.uniforms.uClippingPlane.value.set(
          clippingPlane.normal.x,
          clippingPlane.normal.y,
          clippingPlane.normal.z,
          clippingPlane.constant
        );
      } else {
        // Reset the clipping plane uniform so that no fragments are clipped.
        material.userData.shader.uniforms.uClippingPlane.value.set(0, 0, 0, 0);
      }
    }
  }, [clippingPlane, material]);

  useEffect(() => {
    if (!meshRef.current || !pressures || !pressures[currentTimeIndex]) return;
    const instancedMesh = meshRef.current;
    const currentPressures = pressures[currentTimeIndex];
    const finalPressures = pressures[pressures.length - 1];
    const currentMin = Math.min(...currentPressures);
    const currentMax = Math.max(...currentPressures);
    const finalMin = Math.min(...finalPressures);
    const finalMax = Math.max(...finalPressures);
    const diffMPa = (finalMax - finalMin) / 1e6;
    for (let idx = 0; idx < instanceCount; idx++) {
      const { matrix, i, j } = instanceInfo[idx];
      const pressureVal = currentPressures[i + j * nx];
      const color = pressureToColor(pressureVal, currentMin, currentMax);
      colorAttribute.setXYZ(idx, color.r, color.g, color.b);
      const fractionHeight = (pressureVal - finalMin) / Math.max(1e-30, finalMax - finalMin);
      const zThickness = is3D ? fractionHeight * diffMPa * heightExaggeration : 0.01;
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
   FixedOriginMarker:
   Renders a small white sphere that is attached to the camera.
   This ensures it isn’t translated by panning (right-click dragging).
------------------------------------------------------------------- */
const FixedOriginMarker: React.FC = () => {
  const { camera } = useThree();
  const markerRef = useRef<THREE.Mesh>(null);

  useEffect(() => {
    if (markerRef.current) {
      camera.add(markerRef.current);
    }
    return () => {
      if (markerRef.current) {
        camera.remove(markerRef.current);
      }
    };
  }, [camera]);

  // Position the marker relative to the camera so it stays in the viewport.
  // Adjust the position as needed (e.g., slightly offset from the center).
  return (
    <mesh ref={markerRef} position={[0, 0, -5]}>
      <sphereGeometry args={[0.5, 16, 16]} />
      <meshBasicMaterial color="white" />
    </mesh>
  );
};

/* ------------------------------------------------------------------
   SceneContents:
   Sets up camera, lights, and background.
   Computes the clipping plane from external state (position & rotation)
   and shifts the volume’s vertical center to the world origin.
------------------------------------------------------------------- */
const SceneContents: React.FC<SceneProps> = (props) => {
  const {
    simulationData,
    cameraDistance = 100,
    showClippingPlane,
    clippingPlanePosition = { x: 0, y: 0, z: 0 },
    clippingPlaneRotation = { x: 0, y: 0, z: 0 },
    is3D = true,
    heightExaggeration = 1,
  } = props;
  const { camera, gl, scene } = useThree();

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

  // Compute the clipping plane matrix from external state.
  const planeMatrix = useMemo(() => {
    const pos = new THREE.Vector3(
      clippingPlanePosition.x,
      clippingPlanePosition.y,
      clippingPlanePosition.z
    );
    const euler = new THREE.Euler(
      THREE.MathUtils.degToRad(clippingPlaneRotation.x),
      THREE.MathUtils.degToRad(clippingPlaneRotation.y),
      THREE.MathUtils.degToRad(clippingPlaneRotation.z)
    );
    const quat = new THREE.Quaternion().setFromEuler(euler);
    const m = new THREE.Matrix4();
    m.compose(pos, quat, new THREE.Vector3(1, 1, 1));
    return m;
  }, [clippingPlanePosition, clippingPlaneRotation]);

  // Compute a THREE.Plane from the planeMatrix.
  const clippingPlane = useMemo(() => {
    const localNormal = new THREE.Vector3(1, 0, 0);
    const worldNormal = localNormal
      .clone()
      .applyMatrix4(new THREE.Matrix4().extractRotation(planeMatrix))
      .normalize();
    const worldPos = new THREE.Vector3();
    planeMatrix.decompose(worldPos, new THREE.Quaternion(), new THREE.Vector3());
    const constant = -worldNormal.dot(worldPos);
    return new THREE.Plane(worldNormal, constant);
  }, [planeMatrix]);

  // Compute vertical offset to shift the volume center.
  // Since the instance x,y positions are centered, we only need to adjust z.
  const zOffset = useMemo(() => {
    if (!simulationData || !is3D) return 0;
    const finalPressures = simulationData.pressures[simulationData.pressures.length - 1];
    const finalMin = Math.min(...finalPressures);
    const finalMax = Math.max(...finalPressures);
    const diffMPa = (finalMax - finalMin) / 1e6;
    return (diffMPa * heightExaggeration) / 2;
  }, [simulationData, is3D, heightExaggeration]);

  return (
    <>
      <OrbitControls enableZoom={false} />
      <ambientLight intensity={0.5} />
      <directionalLight position={[20, 30, 50]} intensity={0.4} castShadow />
      {/* Use FixedOriginMarker to keep the origin marker fixed in the view */}
      <FixedOriginMarker />
      {simulationData && (
        // Wrap the reservoir blocks in a group and shift vertically.
        <group position={[0, 0, -zOffset]}>
          <ReservoirBlocksInstanced
            {...props}
            clippingPlane={showClippingPlane ? clippingPlane : undefined}
          />
        </group>
      )}
    </>
  );
};

/* ------------------------------------------------------------------
   Top-level Scene component with Canvas.
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
