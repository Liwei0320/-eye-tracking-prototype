# 多 USB 摄像头简易眼动仪原型

这是一个最小可用的 Python 原型：默认同时读取多个摄像头，优先用 MediaPipe Face Mesh 提取眼部/虹膜特征；如果 MediaPipe 不可用，会退到 OpenCV Haar 眼部检测，通过 9 点标定把眼部特征映射到屏幕注视点。

它适合先验证可行性。普通 USB 摄像头、普通光照下精度有限；如果要更稳定，建议后续换红外摄像头并加 850nm 红外补光。

## 安装

推荐使用 Python 3.10、3.11 或 3.12。Python 3.13/3.14 通常不能使用 MediaPipe 的旧版 FaceMesh API，但可以用 `--backend haar` 跑 OpenCV 备用方案。

```powershell
python -m pip install -r requirements.txt
```

如果运行时报 `cvDestroyAllWindows`、`The function is not implemented`，说明当前 OpenCV 没有窗口显示支持。先卸掉 headless 版本，再装普通版本：

```powershell
python -m pip uninstall -y opencv-python-headless opencv-contrib-python-headless opencv-python
python -m pip install opencv-python numpy
```

## 运行

当前摄像头布局：

- `0`：中间摄像头
- `1`：左边摄像头
- `2`：右边摄像头
- `3`：第四个摄像头（可选）

程序默认会按这个布局全部打开：

```powershell
python dual_usb_eye_tracker.py --cam-left 1 --cam-right 2 --cam-third 0 --cam-fourth 3
```

如果你当前使用的是 Python 3.14，请先这样运行：

```powershell
python dual_usb_eye_tracker.py --cam-left 1 --cam-right 2 --cam-third 0 --cam-fourth 3 --backend haar
```

如果多个摄像头同时开时画面跳动，先降低 USB 带宽压力：

```powershell
python dual_usb_eye_tracker.py --backend haar --width 640 --height 480 --fps 15
```

也可以直接指定任意摄像头索引列表：

```powershell
python dual_usb_eye_tracker.py --backend haar --camera-indexes 1,2,0,3 --width 640 --height 480 --fps 15
```

如果你想临时退回两摄或三摄模式：

```powershell
python dual_usb_eye_tracker.py --backend haar --use-cameras left-right
python dual_usb_eye_tracker.py --backend haar --use-cameras left-right-third
```

如果某个摄像头不稳定，可以单独测试：

```powershell
python dual_usb_eye_tracker.py --backend haar --use-cameras left
python dual_usb_eye_tracker.py --backend haar --use-cameras right
python dual_usb_eye_tracker.py --backend haar --use-cameras third
python dual_usb_eye_tracker.py --backend haar --use-cameras fourth
```

如果只有左边摄像头稳定，先用左边单摄跑通标定：

```powershell
python dual_usb_eye_tracker.py --backend haar --use-cameras left --width 640 --height 480 --fps 15 --fourcc MJPG --windowed
```

如果正式模式下左右/中间画面会消失，但预览模式正常，这是因为正式模式在做眼睛检测；Haar 对侧脸和近距离眼部很敏感，检测不到时标定不会采样。可以只用稳定的摄像头做标定，同时把其他摄像头作为监看画面显示：

```powershell
python dual_usb_eye_tracker.py --backend haar --use-cameras left --monitor-camera-indexes 2,0,3 --width 320 --height 240 --fps 25 --fourcc MJPG --windowed
```

这个模式里：`camera 1` 参与眼动标定，`camera 2` 和 `camera 0` 只显示，不参与计算。

如果多摄同时开时其它画面闪烁，进一步降低带宽：

```powershell
python dual_usb_eye_tracker.py --preview-only --camera-indexes 1,2,0,3 --width 320 --height 240 --fps 10 --fourcc MJPG
python dual_usb_eye_tracker.py --backend haar --camera-indexes 1,2,0,3 --width 320 --height 240 --fps 10 --fourcc MJPG --windowed
```

也可以打开诊断模式，看每个摄像头是否在掉帧：

```powershell
python dual_usb_eye_tracker.py --diagnose-cameras --camera-indexes 1,2,0,3 --width 320 --height 240 --fps 10 --fourcc MJPG
```

诊断里 `failures` 是读帧失败次数，`reused` 是为了防止闪黑而复用上一帧的次数。

如果 `failures=0` 但画面还在闪，通常是自动曝光/自动白平衡在跳。先试锁定曝光：

```powershell
python dual_usb_eye_tracker.py --diagnose-cameras --camera-indexes 1,2,0,3 --width 320 --height 240 --fps 10 --fourcc MJPG --manual-exposure --exposure -6 --gain 0 --manual-white-balance --white-balance 5000
```

如果画面太暗，把 `--exposure -6` 改成 `-5` 或 `-4`；如果太亮，改成 `-7` 或 `-8`。

如果锁曝光后仍然闪，开启软件亮度稳定：

```powershell
python dual_usb_eye_tracker.py --diagnose-cameras --camera-indexes 1,2,0,3 --width 320 --height 240 --fps 25 --fourcc MJPG --normalize-brightness --target-brightness 115 --brightness-alpha 0.08
```

正式运行：

```powershell
python dual_usb_eye_tracker.py --backend haar --camera-indexes 1,2,0,3 --width 320 --height 240 --fps 25 --fourcc MJPG --normalize-brightness --target-brightness 115 --brightness-alpha 0.08 --windowed
```

如果想把 9 点标定做两遍，提高拟合稳定性：

```powershell
python dual_usb_eye_tracker.py --backend haar --use-cameras left-right --monitor-camera-indexes 0,3 --width 320 --height 240 --fps 25 --fourcc MJPG --windowed --calibration-rounds 2
```

正式模式默认会把摄像头画面嵌在标定窗口底部：

```powershell
python dual_usb_eye_tracker.py --backend haar --camera-indexes 1,2,0,3 --width 320 --height 240 --fps 10 --fourcc MJPG --windowed
```

如果想让摄像头画面单独开一个窗口：

```powershell
python dual_usb_eye_tracker.py --backend haar --camera-indexes 1,2,0,3 --width 320 --height 240 --fps 10 --fourcc MJPG --windowed --separate-previews
```

多摄模式需要重新标定，不能直接复用不同摄像头数量下生成的 `calibration.json`。

如果是检测到的眼睛/瞳孔点在跳，而摄像头画面本身稳定，可以加强特征滤波：

```powershell
python dual_usb_eye_tracker.py --backend haar --feature-alpha 0.18 --max-feature-jump 0.12 --gaze-alpha 0.16
```

移动速度和稳定性可以这样调：

```powershell
# 稳定优先，红点移动慢
python dual_usb_eye_tracker.py --backend haar --width 640 --height 480 --fps 15 --windowed --feature-alpha 0.14 --max-feature-jump 0.10 --gaze-alpha 0.12

# 平衡推荐
python dual_usb_eye_tracker.py --backend haar --width 640 --height 480 --fps 15 --windowed --feature-alpha 0.22 --max-feature-jump 0.16 --gaze-alpha 0.22

# 反应更快，但更容易抖
python dual_usb_eye_tracker.py --backend haar --width 640 --height 480 --fps 15 --windowed --feature-alpha 0.45 --max-feature-jump 0.30 --gaze-alpha 0.45
```

- `--feature-alpha`：眼部特征平滑。越小越稳，越大越快。
- `--max-feature-jump`：过滤突然跳变。越小越严格，太小可能卡住。
- `--gaze-alpha`：红点移动平滑。越小红点越慢越稳，越大红点越灵敏。

如果摄像头索引不对，先列出可用摄像头：

```powershell
python dual_usb_eye_tracker.py --list-cameras
```

如果没有摄像头画面，先用只预览模式排查：

```powershell
python dual_usb_eye_tracker.py --preview-only --camera-indexes 1,2,0,3 --width 640 --height 480 --fps 15
```

如果预览窗口还是黑的或不弹出，先保存三张截图判断摄像头是否真的读到了画面：

```powershell
python dual_usb_eye_tracker.py --save-snapshots --camera-indexes 1,2,0 --width 640 --height 480 --fps 15
```

运行后会在当前目录生成 `camera_1_snapshot.jpg`、`camera_2_snapshot.jpg`、`camera_0_snapshot.jpg`。如果图片正常，问题是窗口显示/OpenCV GUI；如果图片也没有，问题是摄像头索引、权限、占用或 USB 连接。

如果还是没有画面，逐个摄像头测试：

```powershell
python dual_usb_eye_tracker.py --preview-only --camera-indexes 0 --width 640 --height 480 --fps 15
python dual_usb_eye_tracker.py --preview-only --camera-indexes 1 --width 640 --height 480 --fps 15
python dual_usb_eye_tracker.py --preview-only --camera-indexes 2 --width 640 --height 480 --fps 15
python dual_usb_eye_tracker.py --preview-only --camera-indexes 3 --width 640 --height 480 --fps 15
```

## 使用流程

1. 固定摄像头，建议左右两个拍眼睛，中间/第三/第四拍完整脸部或补充角度。
2. 运行程序后，进入 9 点标定。
3. 每个白点出现时盯住它，程序会自动采样。
4. 标定完成后进入实时跟踪，屏幕上红点是估计注视点。
5. 按 `q` 或 `Esc` 退出。
6. 标定结果会保存到 `calibration.json`，下次可用：

```powershell
python dual_usb_eye_tracker.py --cam-left 1 --cam-right 2 --cam-third 0 --cam-fourth 3 --load-calibration calibration.json
```

## 注意事项

- 头部尽量不要大幅移动。
- 光照要稳定，避免眼镜强反光。
- 摄像头位置移动后需要重新标定。
- 多个普通 USB 摄像头不同步是正常的，本原型按最近帧做融合。
- 若画面检测不到眼部，先调整摄像头距离、角度和光照。
