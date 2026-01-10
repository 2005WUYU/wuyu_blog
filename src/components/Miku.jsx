// src/components/Miku.jsx
import React, { useEffect, useRef } from 'react';
import * as PIXI from 'pixi.js';
import { Live2DModel } from 'pixi-live2d-display/cubism4';

if (typeof window !== 'undefined') {
  window.PIXI = PIXI;
}

export default function Miku() {
  const canvasRef = useRef(null);
  const appRef = useRef(null);
  const modelRef = useRef(null);

  // 定义画布固定尺寸 (这个尺寸足够放下一个半身/全身的 Live2D)
  const CANVAS_WIDTH = 400;
  const CANVAS_HEIGHT = 500;

  useEffect(() => {
    if (typeof window.Live2DCubismCore === 'undefined') {
      console.error('Live2DCubismCore not loaded!');
      return;
    }

    // 1. 初始化 PIXI (注意：不再使用 window 大小，而是固定大小)
    const app = new PIXI.Application({
      view: canvasRef.current,
      autoStart: true,
      backgroundAlpha: 0,
      width: CANVAS_WIDTH,
      height: CANVAS_HEIGHT,
    });
    appRef.current = app;

    const loadModel = async () => {
      try {
        // 2. 加载模型
        const model = await Live2DModel.from('/live2D/miku/miku_sample_t04.model3.json', {
          autoInteract: false,
        });
        
        modelRef.current = model;
        app.stage.addChild(model);

        // 3. 调整模型大小和位置 (关键！)
        // 坐标系现在是基于 400x500 的画布，而不是全屏
        const scale = 0.3; // 根据模型实际大小微调，Miku通常很大
        model.scale.set(scale);
        
        // 居中放置在画布底部
        model.x = CANVAS_WIDTH / 2;
        model.y = CANVAS_HEIGHT; 
        model.anchor.set(0.5, 1); // 锚点设在模型脚底中心

        if (model.internalModel.motionManager) {
          model.internalModel.motionManager.startRandomMotion('Idle');
        }

        // 4. 鼠标视线跟随 (坐标计算逻辑修正)
        const onMouseMove = (e) => {
          if (model && model.internalModel) {
            const rect = canvasRef.current.getBoundingClientRect();
            // 计算鼠标在画布内的相对坐标
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            model.internalModel.focusController?.focus(
              (x - model.x) / model.width,
              (y - model.y) / model.height
            );
          }
        };
        window.addEventListener('mousemove', onMouseMove);
        model._onMouseMove = onMouseMove;

        // 5. 点击交互逻辑
        const onClick = (e) => {
          const rect = canvasRef.current.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const y = e.clientY - rect.top;
          
          // 简单的判定：只要点在画布内就触发，因为画布本身就很小
          // 如果需要精确判定点在人身上，可以使用 model.hitTest(x,y) 但比较复杂
          if (model.internalModel.motionManager) {
             console.log("Miku Tap!");
             model.internalModel.motionManager.startRandomMotion('Tap');
          }
        };
        // 监听画布的点击，而不是全局点击
        canvasRef.current.addEventListener('click', onClick);
        model._onClick = onClick;

      } catch (error) {
        console.error('Failed to load Live2D model:', error);
      }
    };

    loadModel();

    // 不需要 Resize 监听了，因为我们是固定挂件

    return () => {
      if (modelRef.current) {
        if (modelRef.current._onMouseMove) {
          window.removeEventListener('mousemove', modelRef.current._onMouseMove);
        }
        if (modelRef.current._onClick && canvasRef.current) {
          canvasRef.current.removeEventListener('click', modelRef.current._onClick);
        }
      }
      if (appRef.current) {
        appRef.current.destroy(true, { children: true });
      }
    };
  }, []);

  return (
    <canvas 
      ref={canvasRef} 
      style={{
        position: 'fixed',
        // 关键定位：钉死在右下角
        bottom: 0, 
        right: 0,
        zIndex: 50, // 层级不用太夸张，不挡 Header 就行
        // 关键交互：必须允许点击！
        // 因为我们已经通过 bottom/right 避开了导航栏，所以这里可以开启交互
        pointerEvents: 'auto', 
        width: '400px',
        height: '500px',
      }} 
    />
  );
}