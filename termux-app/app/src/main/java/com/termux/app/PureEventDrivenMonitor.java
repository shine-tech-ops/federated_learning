package com.termux.app;

import android.annotation.SuppressLint;
import android.app.Service;
import android.content.Intent;
import android.os.*;
import android.util.Log;

import com.termux.shared.termux.TermuxConstants;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.*;

public class PureEventDrivenMonitor extends Service {
    private static final String TAG = "PureEventMonitor";

    private FileObserver termuxObserver;
    private Handler eventHandler;
    private ExecutorService contentExecutor;

    // 文件状态缓存（避免重复读取相同内容）
    private final ConcurrentHashMap<String, FileState> fileCache = new ConcurrentHashMap<>();
    private final Object fileCacheLock = new Object();

    static class FileState {
        String contentHash;
        long lastModified;
        volatile boolean isReading = false; // 使用volatile保证可见性

        // 辅助方法：安全地设置读取状态
        synchronized boolean beginReading() {
            if (isReading) {
                return false; // 已经在读取中
            }
            isReading = true;
            return true;
        }

        synchronized void endReading() {
            isReading = false;
        }
    }

    @Override
    public void onCreate() {
        super.onCreate();

        // 处理事件的线程
        HandlerThread handlerThread = new HandlerThread("EventHandler");
        handlerThread.start();
        eventHandler = new Handler(handlerThread.getLooper());

        // 读取文件内容的线程池
        contentExecutor = Executors.newFixedThreadPool(2);

        setupPureEventMonitoring();

        Log.i(TAG, "纯事件驱动监控已启动");
    }

    private void setupPureEventMonitoring() {
        @SuppressLint("SdCardPath") String termuxPath = TermuxConstants.TERMUX_HOME_DIR_PATH + "/transmit";

        termuxObserver = new FileObserver(termuxPath, FileObserver.ALL_EVENTS) {
            @Override
            public void onEvent(int event, String relativePath) {
                if (relativePath == null) return;

                String fullPath = termuxPath + "/" + relativePath;

                // 异步处理事件，避免阻塞事件回调
                eventHandler.post(() -> {
                    processFileEvent(event, fullPath, relativePath);
                });
            }
        };

        termuxObserver.startWatching();
    }

    private void processFileEvent(int event, String fullPath, String relativePath) {
        File file = new File(fullPath);

        // 过滤：只处理文件，不处理目录（除非需要监控目录创建）
        if (file.isDirectory()) {
            handleDirectoryEvent(event, fullPath);
            return;
        }

        // 过滤文件类型
        if (!isTargetFile(relativePath)) {
            return;
        }

        switch (event) {
            case FileObserver.CREATE:
            case FileObserver.MOVED_TO:
                handleFileCreated(fullPath);
                break;

            case FileObserver.MODIFY:
                // MODIFY事件可能触发多次，需要防抖处理
                handleFileModifiedWithDebounce(fullPath);
                break;

            case FileObserver.CLOSE_WRITE:
                // 文件关闭写入是最佳的内容读取时机
                handleFileClosedAfterWrite(fullPath);
                break;

            case FileObserver.DELETE:
            case FileObserver.MOVED_FROM:
                handleFileDeleted(fullPath);
                break;

            case FileObserver.ATTRIB:
                // 属性变化（如权限、时间戳）
                handleFileAttributesChanged(fullPath);
                break;

            default:
                Log.v(TAG, String.format("其他事件: 0x%x -> %s", event, relativePath));
                break;
        }
    }

    private void handleFileCreated(String filePath) {
        // 新文件，立即读取内容
        readFileContentAsync(filePath, true);
    }

    private void handleFileModifiedWithDebounce(String filePath) {
        // 移除之前的防抖消息（使用文件路径作为标识）
        eventHandler.removeCallbacksAndMessages(filePath);

        // 设置防抖延迟（300ms后检查）
        Runnable checkRunnable = () -> checkIfContentChanged(filePath);
        Message msg = Message.obtain(eventHandler, checkRunnable);
        msg.obj = filePath;

        eventHandler.sendMessageDelayed(msg, 300);
    }

    private void handleFileClosedAfterWrite(String filePath) {
        // 文件关闭后写入完成，这是读取内容的最佳时机
        readFileContentAsync(filePath, false);
    }

    private void checkIfContentChanged(String filePath) {
        File file = new File(filePath);
        if (!file.exists()) {
            Log.d(TAG, "检查文件时文件已不存在: " + filePath);
            return;
        }

        FileState cachedState = fileCache.get(filePath);
        long currentModified = file.lastModified();

        // 检查修改时间是否真的变化
        if (cachedState != null && currentModified <= cachedState.lastModified) {
            Log.v(TAG, "文件修改时间未变化: " + filePath);
            return; // 文件实际未修改
        }

        // 文件已修改，读取内容
        Log.d(TAG, "文件修改时间变化，触发读取: " + filePath);
        readFileContentAsync(filePath, false);
    }

    private void readFileContentAsync(String filePath, boolean isNewFile) {
        contentExecutor.submit(() -> {
            try {
                File file = new File(filePath);
                if (!file.exists()) {
                    Log.w(TAG, "读取文件时文件已不存在: " + filePath);
                    return;
                }

                // 检查文件状态，防止并发读取
                FileState state = fileCache.get(filePath);
                if (state == null) {
                    state = new FileState();
                }

                // 尝试开始读取
                if (!state.beginReading()) {
                    Log.v(TAG, "文件正在被其他线程读取，跳过: " + filePath);
                    return;
                }

                try {
                    long modifiedTime = file.lastModified();

                    // 读取文件内容
                    String content = readFileSafely(filePath);
                    String newHash = calculateContentHash(content);

                    // 检查内容是否真正变化
                    boolean contentChanged = state.contentHash == null ||
                        !state.contentHash.equals(newHash);

                    if (contentChanged) {
                        // 更新缓存
                        state.contentHash = newHash;
                        state.lastModified = modifiedTime;
                        fileCache.put(filePath, state);

                        // 通知内容变化
                        notifyContentChanged(filePath, content, modifiedTime, isNewFile);

                        Log.d(TAG, String.format("内容变化: %s (大小: %d, 哈希: %s)",
                            filePath, content.length(), newHash.substring(0, Math.min(8, newHash.length()))));
                    } else {
                        Log.v(TAG, "文件内容未变化: " + filePath);
                    }
                } finally {
                    // 确保释放读取锁
                    state.endReading();
                }

            } catch (Exception e) {
                Log.e(TAG, "读取文件失败: " + filePath, e);

                // 确保异常时也释放读取锁
                FileState state = fileCache.get(filePath);
                if (state != null) {
                    state.endReading();
                }
            }
        });
    }

    private String readFileSafely(String filePath) throws IOException {
        File file = new File(filePath);
        if (!file.exists() || file.length() == 0) {
            return "";
        }

        // 使用FileInputStream，更标准的做法
        try (FileInputStream fis = new FileInputStream(file);
             BufferedInputStream bis = new BufferedInputStream(fis)) {

            long fileSize = file.length();
            int maxReadSize = 1024 * 1024; // 最大读取1MB

            if (fileSize > maxReadSize) {
                // 大文件，只读取部分
                byte[] buffer = new byte[maxReadSize];
                int bytesRead = bis.read(buffer);
                if (bytesRead > 0) {
                    return new String(buffer, 0, bytesRead, StandardCharsets.UTF_8) +
                        "\n...[文件过大，已截断前" + maxReadSize + "字节]";
                }
                return "";
            } else {
                // 小文件，完整读取
                byte[] buffer = new byte[(int) fileSize];
                int bytesRead = bis.read(buffer);
                if (bytesRead > 0) {
                    return new String(buffer, 0, bytesRead, StandardCharsets.UTF_8);
                }
                return "";
            }
        }
    }

    private void handleFileDeleted(String filePath) {
        FileState removed = fileCache.remove(filePath);
        if (removed != null) {
            Log.d(TAG, "文件已删除，从缓存移除: " + filePath);
        }
        notifyFileDeleted(filePath);
    }

    private void handleFileAttributesChanged(String filePath) {
        // 可以检查权限变化等
        File file = new File(filePath);
        if (file.exists()) {
            Log.d(TAG, "文件属性变化: " + filePath +
                " 可读: " + file.canRead() + " 可写: " + file.canWrite() +
                " 最后修改: " + file.lastModified());
        }
    }

    private void handleDirectoryEvent(int event, String dirPath) {
        switch (event) {
            case FileObserver.CREATE:
                // 新目录，可能需要为其设置监控
                Log.d(TAG, "新目录创建: " + dirPath);
                // 这里可以递归设置对新目录的监控
                break;
            case FileObserver.DELETE:
                Log.d(TAG, "目录删除: " + dirPath);
                break;
        }
    }

    private String calculateContentHash(String content) {
        // 使用更稳定的哈希算法
        return Integer.toHexString(content.hashCode());
        // 或者使用更复杂的哈希：
        // return Integer.toHexString(java.util.Arrays.hashCode(content.getBytes(StandardCharsets.UTF_8)));
    }

    private boolean isTargetFile(String filename) {
        if (filename == null || filename.isEmpty()) {
            return false;
        }

        // 可以根据需要扩展
        return filename.endsWith(".txt") ||
            filename.endsWith(".log") ||
            filename.endsWith(".sh") ||
            filename.endsWith(".json") ||
            filename.endsWith(".xml") ||
            filename.endsWith(".conf") ||
            filename.endsWith(".ini") ||
            filename.endsWith(".properties");
    }

    private void notifyContentChanged(String filePath, String content,
                                      long modifiedTime, boolean isNew) {
        // 发送广播（注意：需要注册接收器）
        Intent intent = new Intent("com.termux.PURE_EVENT_CONTENT_CHANGED");
        intent.putExtra("file_path", filePath);
        intent.putExtra("content", content);
        intent.putExtra("modified_time", modifiedTime);
        intent.putExtra("is_new_file", isNew);
        intent.putExtra("content_length", content.length());

        sendBroadcast(intent);

        Log.i(TAG, String.format("通知发送: %s %s 大小: %d",
            filePath, isNew ? "[新文件]" : "[修改]", content.length()));
    }

    private void notifyFileDeleted(String filePath) {
        Intent intent = new Intent("com.termux.PURE_EVENT_FILE_DELETED");
        intent.putExtra("file_path", filePath);
        sendBroadcast(intent);

        Log.i(TAG, "文件删除通知: " + filePath);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null) {
            String action = intent.getAction();
            if ("START".equals(action)) {
                Log.i(TAG, "收到启动命令");
            } else if ("STOP".equals(action)) {
                stopSelf();
            }
        }
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        Log.i(TAG, "服务销毁，清理资源");

        if (termuxObserver != null) {
            termuxObserver.stopWatching();
            termuxObserver = null;
        }

        if (contentExecutor != null) {
            contentExecutor.shutdownNow();
            try {
                if (!contentExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                    Log.w(TAG, "线程池未能在5秒内终止");
                }
            } catch (InterruptedException e) {
                Log.w(TAG, "等待线程池终止时被中断", e);
                Thread.currentThread().interrupt();
            }
        }

        // 清空缓存
        fileCache.clear();

        super.onDestroy();
        Log.i(TAG, "纯事件驱动监控已停止");
    }

    @Override
    public IBinder onBind(Intent intent) {
        // 如果不支持绑定，返回null
        return null;
    }
}
