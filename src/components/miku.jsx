// src/components/Miku.jsx
import React, { useEffect, useRef } from 'react';
import * as PIXI from 'pixi.js';
import { Live2DModel } from 'pixi-live2d-display/cubism4';

// 将 PIXI 暴露给 window（pixi-live2d-display 需要）
if (typeof window !== 'undefined') {
  window.PIXI = PIXI;
}

export default function Miku() {
  const canvasRef = useRef(null);
  const appRef = useRef(null);
  const modelRef = useRef(null);

  useEffect(() => {
    // 检查 Cubism Core 是否已加载
    if (typeof window.Live2DCubismCore === 'undefined') {
      console.error('Live2DCubismCore not loaded!');
      return;
    }

    // 初始化 PIXI 应用
    const app = new PIXI.Application({
      view: canvasRef.current,
      autoStart: true,
      backgroundAlpha: 0,
      width: window.innerWidth,
      height: window.innerHeight,
    });
    appRef.current = app;

    // 加载模型
    const loadModel = async () => {
      try {
        const model = await Live2DModel.from('/live2D/miku/miku_sample_t04.model3.json', {
          autoInteract: false, // 禁用自动交互，避免兼容性问题
        });
        
        modelRef.current = model;
        app.stage.addChild(model);

        // 调整模型大小和位置
        const scale = 0.5;
        model.scale.set(scale);
        
        // 定位到右下角
        model.x = window.innerWidth - model.width / 2 - 50;
        model.y = window.innerHeight - 50;
        model.anchor.set(0.5, 1);

        // 开始播放 idle 动画
        if (model.internalModel.motionManager) {
          model.internalModel.motionManager.startRandomMotion('Idle');
        }

        // 鼠标跟随（手动实现，避免内置交互的兼容性问题）
        const onMouseMove = (e) => {
          if (model && model.internalModel) {
            // 计算鼠标相对于模型的位置
            const rect = canvasRef.current.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // 让模型看向鼠标
            model.internalModel.focusController?.focus(
              (x - model.x) / model.width,
              (y - model.y) / model.height
            );
          }
        };
        
        window.addEventListener('mousemove', onMouseMove);
        model._onMouseMove = onMouseMove;

        // 点击触发动作
        canvasRef.current.style.pointerEvents = 'auto';
        const onClick = (e) => {
          const rect = canvasRef.current.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const y = e.clientY - rect.top;
          
          // 检查点击是否在模型范围内
          const modelBounds = model.getBounds();
          if (x >= modelBounds.x && x <= modelBounds.x + modelBounds.width &&
              y >= modelBounds.y && y <= modelBounds.y + modelBounds.height) {
            // 播放随机 Tap 动画
            if (model.internalModel.motionManager) {
              model.internalModel.motionManager.startRandomMotion('Tap');
            }
          }
        };
        canvasRef.current.addEventListener('click', onClick);
        model._onClick = onClick;

        console.log('Miku model loaded successfully!');
      } catch (error) {
        console.error('Failed to load Live2D model:', error);
      }
    };

    loadModel();

    // 窗口大小变化时重新定位
    const handleResize = () => {
      if (appRef.current) {
        appRef.current.renderer.resize(window.innerWidth, window.innerHeight);
      }
      if (modelRef.current) {
        modelRef.current.x = window.innerWidth - modelRef.current.width / 2 - 50;
        modelRef.current.y = window.innerHeight - 50;
      }
    };
    window.addEventListener('resize', handleResize);

    // 清理函数
    return () => {
      window.removeEventListener('resize', handleResize);
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
        top: 0,
        left: 0,
        zIndex: 999,
        pointerEvents: 'none',
        width: '100vw',
        height: '100vh',
      }} 
    />
  );
}
