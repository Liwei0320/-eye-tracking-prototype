# 眼动仪方案对比报告

日期：2026-06-17  
项目：多 USB 摄像头眼动仪原型

## 1. 对比结论

最新最佳方案为：

```text
四摄 + 无眼镜 + adaptive pupil + cubic + 两轮标定
```

相对历史最好一摄结果，最新四摄方案已经有明确提升：

- RMSE 降低 `27.2 px`，提升约 `13.1%`
- 平均误差降低 `35.4 px`，提升约 `19.9%`
- 检测率同为 `100%`
- P95 误差几乎持平，四摄仅差 `2.1 px`

一句话结论：

> 最新四摄方案在平均精度上已经超过一摄，稳定性基本持平，是当前推荐方案。

## 2. 一摄 vs 四摄

| 指标 | 一摄最好历史结果 | 最新四摄最佳结果 | 对比 |
|---|---:|---:|---:|
| 验证文件 | `validation_left_20260611_154501.json` | `validation_all_20260615_162421.json` | - |
| 标定文件 | `calibration_left_trial2.json` | `calibration_no_glasses_cubic_round2.json` | - |
| 配置 | `left` | `all` | 四摄 |
| RMSE | 207.9 px | 180.7 px | 四摄好 27.2 px |
| 平均误差 | 178.3 px | 142.9 px | 四摄好 35.4 px |
| P95 误差 | 340.4 px | 342.5 px | 一摄略好 2.1 px |
| 检测率 | 100.0% | 100.0% | 持平 |
| 有效样本 | 500/500 | 500/500 | 持平 |

## 3. 四摄方案演进

| 方案 | 标定文件 | RMSE | 平均误差 | P95 | 检测率 | 结论 |
|---|---|---:|---:|---:|---:|---|
| 戴眼镜/普通二次优化 | `calibration_adaptive_pupil.json` | 189.3 px | 153.3 px | 356.5 px | 97.8% | 精度好，但检测率略低 |
| 无眼镜/单轮二次 | `calibration_no_glasses.json` | 334.1 px | 315.6 px | 493.2 px | 100.0% | 检测好，映射偏差大 |
| 无眼镜/两轮二次 | `calibration_no_glasses_round2.json` | 314.4 px | 246.6 px | 730.4 px | 100.0% | 有改善，但 P95 差 |
| 无眼镜/两轮 cubic | `calibration_no_glasses_cubic_round2.json` | 180.7 px | 142.9 px | 342.5 px | 100.0% | 当前最佳 |

关键发现：

1. `--adaptive-pupil` 是最有效的检测增强参数。
2. 摘眼镜后检测率明显提升，最佳结果达到 `100%`。
3. 只摘眼镜不一定提升精度，必须配合更合适的映射模型。
4. `--cubic` 在两轮标定、多样本条件下效果最好。
5. 线性标定、过严的 `min-active-cameras=3`、低带宽采集、median fusion 都没有超过当前最佳方案。

## 4. 最新四摄逐点表现

验证文件：`validation_all_20260615_162421.json`

| 点位 | 坐标 | 平均误差 | RMSE | P95 | 检测率 |
|---|---:|---:|---:|---:|---:|
| p0 | (480, 324) | 190.4 px | 224.6 px | 312.4 px | 100.0% |
| p1 | (1440, 324) | 187.8 px | 236.8 px | 494.2 px | 100.0% |
| p2 | (960, 540) | 117.1 px | 166.3 px | 370.4 px | 100.0% |
| p3 | (480, 756) | 110.5 px | 127.0 px | 217.8 px | 100.0% |
| p4 | (1440, 756) | 108.8 px | 114.2 px | 151.8 px | 100.0% |

残余问题：

- 上方点位误差仍偏大。
- p1 的 P95 误差最高，说明右上区域仍有偶发偏差。

## 5. 当前算法对比

当前最佳不是深度学习方案，而是传统算法组合：

| 模块 | 当前方案 |
|---|---|
| 检测后端 | OpenCV Haar |
| 瞳孔增强 | adaptive pupil |
| 多摄融合 | partial fusion，至少 2 路有效 |
| 标定拟合 | Ridge regression |
| 异常抑制 | Huber robust fit |
| 映射模型 | cubic 三次多项式 |
| 标定策略 | 两轮 9 点标定，每点 60 样本 |

## 6. 消费级水平判断

当前最佳四摄：

```text
平均误差: 142.9 px，约 3.8°
RMSE: 180.7 px，约 4.8°
检测率: 100%
```

消费级参考：

| 设备/水平 | 典型精度 |
|---|---:|
| Tobii Eye Tracker 5 独立评测 | 约 0.74° |
| Tobii EyeX 论文评测 | <0.6° |
| Pupil Labs Neon | 1.3°-1.8° |
| 当前四摄原型 | 平均约 3.8°，RMSE 约 4.8° |

判断：

- 检测率已经达到可用水平。
- 精度还没有达到成熟消费级产品水平。
- 当前更接近“消费级原型”，适合粗略交互和区域判断。
- 若要接近成熟消费级，需要红外补光、角膜反光点、头动补偿和更强眼部模型。

## 7. 推荐使用方案

运行当前最佳四摄：

```powershell
python "2026-05-13/usb/dual_usb_eye_tracker.py" --backend haar --use-cameras all --camera-indexes 1,2,0,3 --partial-fusion --min-active-cameras 2 --adaptive-pupil --load-calibration calibration_no_glasses_cubic_round2.json --windowed
```

重新标定当前最佳四摄：

```powershell
python "2026-05-13/usb/dual_usb_eye_tracker.py" --backend haar --use-cameras all --camera-indexes 1,2,0,3 --partial-fusion --min-active-cameras 2 --adaptive-pupil --adaptive-ridge --robust-fit --cubic --calibration-rounds 2 --samples-per-point 60 --calibration calibration_no_glasses_cubic_round2.json --validate --windowed
```

## 8. 下一步建议

优先级从高到低：

1. 引入 850nm 红外补光和黑白全局快门摄像头。
2. 从 Haar 替换到 MediaPipe 或专用眼部关键点模型。
3. 增加摄像头畸变校正和头姿补偿。
4. 增加 13 点、16 点或 25 点验证网格。
5. 做在线漂移校正，降低长时间使用偏差。

