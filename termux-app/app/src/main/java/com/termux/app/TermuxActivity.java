package com.termux.app;

import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.content.ActivityNotFoundException;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.ServiceConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.util.Log;
import android.view.ContextMenu;
import android.view.ContextMenu.ContextMenuInfo;
import android.view.Gravity;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.view.inputmethod.InputMethodManager;
import android.webkit.JavascriptInterface;
import android.webkit.JsResult;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.RelativeLayout;
import android.widget.Toast;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.termux.R;
import com.termux.app.api.file.FileReceiverActivity;
import com.termux.app.terminal.TermuxActivityRootView;
import com.termux.app.terminal.TermuxTerminalSessionActivityClient;
import com.termux.app.terminal.io.TermuxTerminalExtraKeys;
import com.termux.shared.activities.ReportActivity;
import com.termux.shared.activity.ActivityUtils;
import com.termux.shared.activity.media.AppCompatActivityUtils;
import com.termux.shared.data.IntentUtils;
import com.termux.shared.android.PermissionUtils;
import com.termux.shared.data.DataUtils;
import com.termux.shared.termux.TermuxConstants;
import com.termux.shared.termux.TermuxConstants.TERMUX_APP.TERMUX_ACTIVITY;
import com.termux.app.activities.HelpActivity;
import com.termux.app.activities.SettingsActivity;
import com.termux.shared.termux.crash.TermuxCrashUtils;
import com.termux.shared.termux.settings.preferences.TermuxAppSharedPreferences;
import com.termux.app.terminal.TermuxSessionsListViewController;
import com.termux.app.terminal.io.TerminalToolbarViewPager;
import com.termux.app.terminal.TermuxTerminalViewClient;
import com.termux.shared.termux.extrakeys.ExtraKeysView;
import com.termux.shared.termux.interact.TextInputDialogUtils;
import com.termux.shared.logger.Logger;
import com.termux.shared.termux.TermuxUtils;
import com.termux.shared.termux.settings.properties.TermuxAppSharedProperties;
import com.termux.shared.termux.theme.TermuxThemeUtils;
import com.termux.shared.theme.NightMode;
import com.termux.shared.view.ViewUtils;
import com.termux.terminal.TerminalSession;
import com.termux.terminal.TerminalSessionClient;
import com.termux.view.TerminalView;
import com.termux.view.TerminalViewClient;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.drawerlayout.widget.DrawerLayout;
import androidx.viewpager.widget.ViewPager;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Properties;

/**
 * A terminal emulator activity.
 * <p/>
 * See
 * <ul>
 * <li>http://www.mongrel-phones.com.au/default/how_to_make_a_local_service_and_bind_to_it_in_android</li>
 * <li>https://code.google.com/p/android/issues/detail?id=6426</li>
 * </ul>
 * about memory leaks.
 */
public final class TermuxActivity extends AppCompatActivity implements ServiceConnection {

    /**
     * The connection to the {@link TermuxService}. Requested in {@link #onCreate(Bundle)} with a call to
     * {@link #bindService(Intent, ServiceConnection, int)}, and obtained and stored in
     * {@link #onServiceConnected(ComponentName, IBinder)}.
     */
    TermuxService mTermuxService;

    /**
     * The {@link TerminalView} shown in  {@link TermuxActivity} that displays the terminal.
     */
    TerminalView mTerminalView;

    /**
     *  The {@link TerminalViewClient} interface implementation to allow for communication between
     *  {@link TerminalView} and {@link TermuxActivity}.
     */
    TermuxTerminalViewClient mTermuxTerminalViewClient;

    /**
     *  The {@link TerminalSessionClient} interface implementation to allow for communication between
     *  {@link TerminalSession} and {@link TermuxActivity}.
     */
    TermuxTerminalSessionActivityClient mTermuxTerminalSessionActivityClient;

    /**
     * Termux app shared preferences manager.
     */
    private TermuxAppSharedPreferences mPreferences;

    /**
     * Termux app SharedProperties loaded from termux.properties
     */
    private TermuxAppSharedProperties mProperties;

    /**
     * The root view of the {@link TermuxActivity}.
     */
    TermuxActivityRootView mTermuxActivityRootView;

    /**
     * The space at the bottom of {@link @mTermuxActivityRootView} of the {@link TermuxActivity}.
     */
    View mTermuxActivityBottomSpaceView;

    /**
     * The terminal extra keys view.
     */
    ExtraKeysView mExtraKeysView;

    /**
     * The client for the {@link #mExtraKeysView}.
     */
    TermuxTerminalExtraKeys mTermuxTerminalExtraKeys;

    /**
     * The termux sessions list controller.
     */
    TermuxSessionsListViewController mTermuxSessionListViewController;

    /**
     * The {@link TermuxActivity} broadcast receiver for various things like terminal style configuration changes.
     */
    private final BroadcastReceiver mTermuxActivityBroadcastReceiver = new TermuxActivityBroadcastReceiver();

    /**
     * The last toast shown, used cancel current toast before showing new in {@link #showToast(String, boolean)}.
     */
    Toast mLastToast;

    /**
     * If between onResume() and onStop(). Note that only one session is in the foreground of the terminal view at the
     * time, so if the session causing a change is not in the foreground it should probably be treated as background.
     */
    private boolean mIsVisible;

    /**
     * If onResume() was called after onCreate().
     */
    private boolean mIsOnResumeAfterOnCreate = false;

    /**
     * If activity was restarted like due to call to {@link #recreate()} after receiving
     * {@link TERMUX_ACTIVITY#ACTION_RELOAD_STYLE}, system dark night mode was changed or activity
     * was killed by android.
     */
    private boolean mIsActivityRecreated = false;

    /**
     * The {@link TermuxActivity} is in an invalid state and must not be run.
     */
    private boolean mIsInvalidState;

    private int mNavBarHeight;

    private float mTerminalToolbarDefaultHeight;


    private static final int CONTEXT_MENU_SELECT_URL_ID = 0;
    private static final int CONTEXT_MENU_SHARE_TRANSCRIPT_ID = 1;
    private static final int CONTEXT_MENU_SHARE_SELECTED_TEXT = 10;
    private static final int CONTEXT_MENU_AUTOFILL_USERNAME = 11;
    private static final int CONTEXT_MENU_AUTOFILL_PASSWORD = 2;
    private static final int CONTEXT_MENU_RESET_TERMINAL_ID = 3;
    private static final int CONTEXT_MENU_KILL_PROCESS_ID = 4;
    private static final int CONTEXT_MENU_STYLING_ID = 5;
    private static final int CONTEXT_MENU_TOGGLE_KEEP_SCREEN_ON = 6;
    private static final int CONTEXT_MENU_HELP_ID = 7;
    private static final int CONTEXT_MENU_SETTINGS_ID = 8;
    private static final int CONTEXT_MENU_REPORT_ID = 9;

    private static final String ARG_TERMINAL_TOOLBAR_TEXT_INPUT = "terminal_toolbar_text_input";
    private static final String ARG_ACTIVITY_RECREATED = "activity_recreated";

    private static final String LOG_TAG = "TermuxActivity";

    private View mFullscreenDialog;
    private FloatingActionButton mFloatingActionButton;

    private WebView webView;

    /**
     * 复制文件或目录从 assets 到目标位置
     */
    private void copyFromAssets(String assetPath, File destDir, boolean overwrite) {
        try {
            // 检查 assets 中是文件还是目录
            InputStream testStream = null;
            boolean isDirectory = false;

            try {
                testStream = getAssets().open(assetPath);
                // 如果能打开，说明是文件
            } catch (IOException e) {
                // 如果是目录，open 会抛出异常，尝试列出目录内容
                try {
                    String[] files = getAssets().list(assetPath);
                    if (files != null && files.length > 0) {
                        isDirectory = true;
                    }
                } catch (IOException e2) {
                    Logger.logError(LOG_TAG, "Asset path not found: " + assetPath);
                    return;
                }
            } finally {
                if (testStream != null) {
                    try {
                        testStream.close();
                    } catch (IOException e) {
                        // 忽略关闭异常
                    }
                }
            }

            if (isDirectory) {
                copyDirectoryFromAssets(assetPath, destDir, overwrite);
            } else {
                copyFileFromAssets(assetPath, destDir, overwrite);
            }

        } catch (Exception e) {
            Logger.logError(LOG_TAG, "Error determining asset type for " + assetPath + ": " + e.getMessage());
        }
    }

    /**
     * 复制文件从 assets 到目标目录
     */
    private void copyFileFromAssets(String assetFileName, File destDir, boolean overwrite) {
        File destFile = new File(destDir, assetFileName);

        // 如果目标文件已存在且不覆盖，则跳过复制
        if (destFile.exists() && !overwrite) {
            Logger.logVerbose(LOG_TAG, assetFileName + " already exists, skipping copy");
            return;
        }

        // 确保目标目录存在
        if (!destDir.exists()) {
            if (!destDir.mkdirs()) {
                Logger.logError(LOG_TAG, "Failed to create directory: " + destDir.getAbsolutePath());
                return;
            }
        }

        // 从 assets 复制文件
        copySingleFileFromAssets(assetFileName, destFile);
    }

    /**
     * 复制目录从 assets 到目标位置
     */
    private void copyDirectoryFromAssets(String assetDirPath, File destDir, boolean overwrite) {
        try {
            // 确保目标目录存在
            if (!destDir.exists()) {
                if (!destDir.mkdirs()) {
                    Logger.logError(LOG_TAG, "Failed to create directory: " + destDir.getAbsolutePath());
                    return;
                }
            }

            // 列出 assets 目录中的所有文件
            String[] files = getAssets().list(assetDirPath);
            if (files == null || files.length == 0) {
                Logger.logVerbose(LOG_TAG, "Empty directory: " + assetDirPath);
                return;
            }

            for (String file : files) {
                String assetFilePath = assetDirPath + "/" + file;
                File destFile = new File(destDir, file);

                // 递归检查是文件还是目录
                InputStream testStream = null;
                boolean isSubDirectory = false;

                try {
                    testStream = getAssets().open(assetFilePath);
                    // 如果能打开，说明是文件
                } catch (IOException e) {
                    // 尝试列出子目录
                    try {
                        String[] subFiles = getAssets().list(assetFilePath);
                        if (subFiles != null && subFiles.length > 0) {
                            isSubDirectory = true;
                        }
                    } catch (IOException e2) {
                        // 既不是文件也不是目录，跳过
                        continue;
                    }
                } finally {
                    if (testStream != null) {
                        try {
                            testStream.close();
                        } catch (IOException e) {
                            // 忽略关闭异常
                        }
                    }
                }

                if (isSubDirectory) {
                    // 递归复制子目录
                    copyDirectoryFromAssets(assetFilePath, destFile, overwrite);
                } else {
                    // 复制文件
                    copySingleFileFromAssets(assetFilePath, destFile, overwrite);
                }
            }

            Logger.logVerbose(LOG_TAG, "Successfully copied directory " + assetDirPath + " to " + destDir.getAbsolutePath());

        } catch (IOException e) {
            Logger.logError(LOG_TAG, "Error copying directory " + assetDirPath + " from assets: " + e.getMessage());
        }
    }

    /**
     * 复制单个文件从 assets 到指定文件位置
     */
    private void copySingleFileFromAssets(String assetFilePath, File destFile) {
        copySingleFileFromAssets(assetFilePath, destFile, true);
    }

    /**
     * 复制单个文件从 assets 到指定文件位置
     */
    private void copySingleFileFromAssets(String assetFilePath, File destFile, boolean overwrite) {
        if (destFile.exists() && !overwrite) {
            return;
        }

        // 确保父目录存在
        File parentDir = destFile.getParentFile();
        assert parentDir != null;
        if (!parentDir.exists()) {
            if (!parentDir.mkdirs()) {
                Logger.logError(LOG_TAG, "Failed to create parent directory: " + parentDir.getAbsolutePath());
                return;
            }
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            try (InputStream in = getAssets().open(assetFilePath);
                 OutputStream out = Files.newOutputStream(destFile.toPath())) {

                byte[] buffer = new byte[1024];
                int length;
                while ((length = in.read(buffer)) > 0) {
                    out.write(buffer, 0, length);
                }

                // 根据文件扩展名判断是否设置为可执行
                if (isExecutableFile(destFile.getName())) {
                    setFileExecutable(destFile);
                }

                Logger.logVerbose(LOG_TAG, "Successfully copied " + assetFilePath + " to " + destFile.getAbsolutePath());

            } catch (IOException e) {
                Logger.logError(LOG_TAG, "Error copying " + assetFilePath + " from assets: " + e.getMessage());
            }
        }
    }

    /**
     * 判断文件是否应该设置为可执行
     */
    private boolean isExecutableFile(String fileName) {
        String[] executableExtensions = {".sh", ".bash", ".py", ".pl", ".rb", ".exe", ""};
        for (String ext : executableExtensions) {
            if (fileName.endsWith(ext)) {
                return true;
            }
        }
        return false;
    }

    /**
     * 设置文件为可执行权限
     */
    private void setFileExecutable(File file) {
        if (file.setExecutable(true)) {
            Logger.logVerbose(LOG_TAG, "Set executable permission for termux-login.sh");
        } else {
            Logger.logError(LOG_TAG, "Failed to set executable permission for termux-login.sh");
        }
    }

    /**
     * 设置全屏弹窗和浮动按钮
     */
    private void setFullscreenDialogAndFloatingButton() {
        // 初始化全屏弹窗
        mFullscreenDialog = findViewById(R.id.fullscreen_dialog);

        // 点击弹窗背景也可以关闭
        mFullscreenDialog.setOnClickListener(v -> {
            hideFullscreenDialog();
        });

        // 初始化浮动按钮
        mFloatingActionButton = findViewById(R.id.floating_action_button);
        mFloatingActionButton.setOnClickListener(v -> {
            showFullscreenDialog();
        });
    }

    // 显示全屏弹窗的方法
    private void showFullscreenDialog() {
        runOnUiThread(() -> {
            if (mFullscreenDialog != null) {
                // 禁用过渡动画
                overridePendingTransition(0, 0);
                mFullscreenDialog.setVisibility(View.VISIBLE);
                mFloatingActionButton.setVisibility(View.GONE);

                // 隐藏软键盘的正确方式
                if (mTermuxTerminalViewClient != null) {
                    mTerminalView.requestFocus();
                }

                // 或者直接使用 InputMethodManager
                hideSoftKeyboard();
            }
        });
    }

    // 隐藏软键盘的方法
    private void hideSoftKeyboard() {
        InputMethodManager imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
        if (imm != null && mTerminalView != null) {
            imm.hideSoftInputFromWindow(mTerminalView.getWindowToken(), 0);
        }
    }

    // 隐藏全屏弹窗的方法
    private void hideFullscreenDialog() {
        runOnUiThread(() -> {
            if (mFullscreenDialog != null) {
                // 禁用过渡动画
                overridePendingTransition(0, 0);
                mFullscreenDialog.setVisibility(View.GONE);
                mFloatingActionButton.setVisibility(View.VISIBLE);
            }
        });
    }

    public void executeTerminalCommand(String command) {
        Logger.logDebug(LOG_TAG, "executeTerminalCommand");
        if (mTermuxService == null) {
            Logger.logError(LOG_TAG, "TermuxService is not available");
            showToast("Termux服务未就绪", true);
            return;
        }

        TerminalSession currentSession = getCurrentSession();
        if (currentSession == null) {
            Logger.logError(LOG_TAG, "No active terminal session");
            showToast("没有活动的终端会话", true);
            return;
        }

        if (currentSession.isRunning()) {
            // 在命令后添加换行符来执行
            String commandToExecute = command.endsWith("\n") ? command : command + "\n";
            currentSession.write(commandToExecute);
            Logger.logDebug(LOG_TAG, "Executing command: " + command);
        } else {
            Logger.logError(LOG_TAG, "Current session is not running");
            showToast("当前会话未运行", true);
        }
    }

    public void executeCommandAfterDelay(String command, long delayMillis) {
        if (command == null || command.isEmpty()) {
            Logger.logError(LOG_TAG, "Command is null or empty");
            return;
        }

        new Handler().postDelayed(() -> {
            executeTerminalCommand(command);
            Logger.logDebug(LOG_TAG, "Executed command after delay: " + command);
        }, delayMillis);
    }

    public void permanentlyDisableExtraKeys() {
        try {
            // Termux 属性文件路径
            File propertiesFile = TermuxConstants.TERMUX_PROPERTIES_PRIMARY_FILE;

            // 读取现有属性
            Properties properties = new Properties();
            if (propertiesFile.exists()) {
                try (FileInputStream fis = new FileInputStream(propertiesFile)) {
                    properties.load(fis);
                }
            }

            // 启动时隐藏键盘
            properties.setProperty("hide-soft-keyboard-on-startup", "true");
            properties.setProperty("extra-keys", "[['DRAWER','ESC','-','HOME','UP','END','PGUP'],['TAB','CTRL','ALT','LEFT','DOWN','RIGHT','PGDN']]");

            // 写回文件
            try (FileOutputStream fos = new FileOutputStream(propertiesFile)) {
                properties.store(fos, "Modified by TermuxActivity");
            }

            showToast("Set Properties Done", true);
            Logger.logDebug(LOG_TAG, "Set Properties Done");

        } catch (Exception e) {
            Logger.logError(LOG_TAG, "Set Properties Fail: " + e.getMessage());
            showToast("Set Properties Fail", true);
        }
    }

    public static boolean downloadFile(String fileURL, String token, String saveDir, String fileName)
        throws IOException {

        // 创建保存目录
        File directory = new File(saveDir);
        if (!directory.exists()) {
            if (!directory.mkdirs()) {
                throw new IOException("无法创建目录: " + saveDir);
            }
        }

        // 完整的文件路径
        String filePath = saveDir + File.separator + fileName;

        // 创建URL连接
        URL url = new URL(fileURL);
        HttpURLConnection httpConn = (HttpURLConnection) url.openConnection();
        httpConn.setRequestMethod("GET");
        httpConn.setConnectTimeout(15000);  // 15秒连接超时
        httpConn.setReadTimeout(15000);     // 15秒读取超时
        httpConn.setRequestProperty("Authorization", "Bearer " + token);

        // 检查响应码
        int responseCode = httpConn.getResponseCode();
        if (responseCode != HttpURLConnection.HTTP_OK) {
            throw new IOException("HTTP错误代码: " + responseCode);
        }

        // 获取文件大小（如果有）
        String contentLength = httpConn.getHeaderField("Content-Length");
        long fileSize = contentLength != null ? Long.parseLong(contentLength) : 0;

        // 创建输入流和输出流
        try (InputStream inputStream = httpConn.getInputStream();
             FileOutputStream outputStream = new FileOutputStream(filePath)) {

            byte[] buffer = new byte[4096];
            int bytesRead;
            long totalBytesRead = 0;

            System.out.println("开始下载文件...");
            if (fileSize > 0) {
                System.out.println("文件大小: " + formatFileSize(fileSize));
            }

            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
                totalBytesRead += bytesRead;

                // 显示下载进度（可选）
                if (fileSize > 0) {
                    double progress = (double) totalBytesRead / fileSize * 100;
                    System.out.printf("下载进度: %.2f%%\r", progress);
                }
            }

            System.out.println("\n下载完成！");
            return true;
        } finally {
            httpConn.disconnect();
        }
    }

    @SuppressLint("DefaultLocale")
    private static String formatFileSize(long size) {
        if (size < 1024) return size + " B";
        if (size < 1024 * 1024) return String.format("%.2f KB", size / 1024.0);
        if (size < 1024 * 1024 * 1024) return String.format("%.2f MB", size / (1024.0 * 1024));
        return String.format("%.2f GB", size / (1024.0 * 1024 * 1024));
    }

    private final BroadcastReceiver fileChangeReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            if ("com.termux.PURE_EVENT_CONTENT_CHANGED".equals(intent.getAction())) {
                String filePath = intent.getStringExtra("file_path");
//                String content = intent.getStringExtra("content");
//                long modifiedTime = intent.getLongExtra("modified_time", 0);
//                boolean isNew = intent.getBooleanExtra("is_new_file", false);

                // 处理文件内容
                Log.i("Receiver", "文件变化1：" + filePath);
//                Log.i("Receiver", "文件变化2：" + content);
//                Log.i("Receiver", "文件变化3：" + modifiedTime);
//                Log.i("Receiver", "文件变化4：" + isNew);

                if (filePath != null) {
                    File file = new File(filePath);
                    String fileName = file.getName();

                    switch (fileName) {
                        // 传递设备信息
                        case "device_info.json": {
                            Log.i("Receiver", "传递设备信息：");
                            String content = intent.getStringExtra("content");
                            if (content != null) {
                                try {
                                    // 先将字符串转为 JSONObject，再转回字符串确保格式正确
                                    JSONObject jsonObject = new JSONObject(content);
                                    String escapedContent = jsonObject.toString()
                                        .replace("'", "\\'");

                                    webView.post(() -> {
                                        webView.evaluateJavascript(
                                            "javascript:onDeviceInfoReceived('" + escapedContent + "')",
                                            null
                                        );
                                    });
                                } catch (JSONException e) {
                                    Log.e("Receiver", "JSON 解析错误: " + e.getMessage());
                                }
                            }
                            break;
                        }
                        // 传递训练任务
                        case "tran_task.json": {
                            Log.i("Receiver", "传递训练任务");
                            break;
                        }
                        // 传递对话历史
                        case "chat_history.json": {
                            Log.i("Receiver", "传递对话历史");
                            break;
                        }
                        // 默认跳过处理
                        default: {
                            Log.i("Receiver", "跳过处理");
                        }
                    }
                }
            }
        }
    };

    public class WebAppInterface {
        private final Context context;
        private final Handler mainHandler;

        public WebAppInterface(Context context) {
            this.context = context;
            this.mainHandler = new Handler(Looper.getMainLooper());
        }

        @JavascriptInterface
        public void closeWindow() {
            // 确保在 UI 线程中执行
            mainHandler.post(() -> {
                if (context instanceof TermuxActivity) {
                    ((TermuxActivity) context).hideFullscreenDialog();
                }
            });
        }

        @JavascriptInterface
        public void showToast(String message) {
            Toast.makeText(context, message, Toast.LENGTH_SHORT).show();
        }

        @JavascriptInterface
        public void getPhoneDeviceInfo() {
            String deviceInfo = "Device: " + Build.MODEL + ", Android: " + Build.VERSION.RELEASE;
            Log.i("TERMUX", deviceInfo);
            // 回调给JavaScript
            webView.post(() -> {
                webView.evaluateJavascript("javascript:onPhoneDeviceInfoReceived('" + deviceInfo + "')", null);
            });
        }

        @JavascriptInterface
        public void syncModel(String url, String authToken, String modelName, String id, String fullFileName) {
            String fileURL = url + "/learn_management/model_version/" + id + "/download/";

            String[] parts = fullFileName.split("[/\\\\]");
            String fileName = parts[parts.length - 1];
            System.out.println("文件名: " + fileName); // 输出: 1765514993222_demo2.pt
            String directory = "";
            if (parts.length > 1) {
                // 重建目录路径
                StringBuilder dirBuilder = new StringBuilder();
                for (int i = 0; i < parts.length - 1; i++) {
                    if (i > 0) {
                        dirBuilder.append("/");
                    }
                    dirBuilder.append(parts[i]);
                }
                directory = dirBuilder.toString();
            }

            String saveFile = TermuxConstants.TERMUX_HOME_DIR_PATH + File.separator + fullFileName;
            String saveDir = TermuxConstants.TERMUX_HOME_DIR_PATH + File.separator + directory;
            System.out.println(url);
            System.out.println(id);
            System.out.println(fullFileName);
            try {
                boolean finish = downloadFile(fileURL, authToken, saveDir, fileName);
                System.out.println("文件下载成功！");
                // 添加模型到 Ollama
                if (finish) {
                    String command = "proot-distro login ubuntu -- bash -c '"
                        + "mkdir -p ~/models/" + modelName + " && "
                        + "cat > ~/models/" + modelName + "/Modelfile << \"EOF\"\n"
                        + "FROM " + modelName + "\n"
                        + "ADAPTER " + saveFile + "\n"
                        + "EOF\n"
                        + "ollama create " + modelName + " -f ~/models/" + modelName + "/Modelfile'";
                    System.out.println(command);
                    executeTerminalCommand(command);
                    // 回调给JavaScript
                    webView.post(() -> {
                        webView.evaluateJavascript("javascript:onSyncModelReceived('文件下载成功！')", null);
                    });
                }
            } catch (IOException e) {
                System.out.println("下载失败: " + e.getMessage());
                // 回调给JavaScript
                webView.post(() -> {
                    webView.evaluateJavascript("javascript:onSyncModelReceived('下载失败: '" + e.getMessage() + ")", null);
                });
            }
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    private void setupWebView() {
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);
        settings.setAllowFileAccessFromFileURLs(true);
        settings.setAllowUniversalAccessFromFileURLs(true);

        // 禁用混合内容检测（如果同时加载http和file内容）
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);

        webView.loadUrl("file:///android_asset/sample_en.html");

        // 添加JavaScript接口
        webView.addJavascriptInterface(new WebAppInterface(this), "Android");
    }

    @SuppressLint({"SetJavaScriptEnabled", "UnspecifiedRegisterReceiverFlag"})
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Logger.logDebug(LOG_TAG, "onCreate");
        mIsOnResumeAfterOnCreate = true;

        if (savedInstanceState != null)
            mIsActivityRecreated = savedInstanceState.getBoolean(ARG_ACTIVITY_RECREATED, false);

        // Delete ReportInfo serialized object files from cache older than 14 days
        ReportActivity.deleteReportInfoFilesOlderThanXDays(this, 14, false);

        // Load Termux app SharedProperties from disk
        mProperties = TermuxAppSharedProperties.getProperties();
        permanentlyDisableExtraKeys();
        reloadProperties();

        setActivityTheme();

        // 复制 termux-login.sh 文件
        File etcDir = new File(TermuxConstants.TERMUX_PREFIX_DIR, "etc");
        copyFileFromAssets("termux-login.sh", etcDir, true); // 总是覆盖
        //
        File homeDir = new File(TermuxConstants.TERMUX_HOME_DIR_PATH);
        // 复制 start.sh
        copyFileFromAssets("start.sh", homeDir, true);
        // 复制 install.sh 文件
        copyFileFromAssets("install.sh", homeDir, true);
        // 复制 login.sh 文件
        copyFileFromAssets("login.sh", homeDir, true);
        // 复制 transmit 文件
        File transmitDir = new File(TermuxConstants.TERMUX_HOME_DIR_PATH, "transmit");
        copyFromAssets("transmit", transmitDir, true);
        // 复制 train_example_1 Python文件
        File trainExampleDir1 = new File(TermuxConstants.TERMUX_HOME_DIR_PATH, "train_example_1");
        copyFromAssets("train_example_1", trainExampleDir1, true);
        // 复制 train_example_2 Python文件
        // File trainExampleDir2 = new File(TermuxConstants.TERMUX_HOME_DIR_PATH, "train_example_2");
        // copyFromAssets("train_example_2", trainExampleDir2, true);

        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_termux);

        // 打开全屏弹窗
        setFullscreenDialogAndFloatingButton();

        // Load termux shared preferences
        // This will also fail if TermuxConstants.TERMUX_PACKAGE_NAME does not equal applicationId
        mPreferences = TermuxAppSharedPreferences.build(this, true);
        if (mPreferences == null) {
            // An AlertDialog should have shown to kill the app, so we don't continue running activity code
            mIsInvalidState = true;
            return;
        }

        setMargins();

        mTermuxActivityRootView = findViewById(R.id.activity_termux_root_view);
        mTermuxActivityRootView.setActivity(this);
        mTermuxActivityBottomSpaceView = findViewById(R.id.activity_termux_bottom_space_view);
        mTermuxActivityRootView.setOnApplyWindowInsetsListener(new TermuxActivityRootView.WindowInsetsListener());

        View content = findViewById(android.R.id.content);
        content.setOnApplyWindowInsetsListener((v, insets) -> {
            mNavBarHeight = insets.getSystemWindowInsetBottom();
            return insets;
        });

        if (mProperties.isUsingFullScreen()) {
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
        }

        setTermuxTerminalViewAndClients();

        setTerminalToolbarView(savedInstanceState);

        setSettingsButtonView();

        setNewSessionButtonView();

        setToggleKeyboardView();

        registerForContextMenu(mTerminalView);

        FileReceiverActivity.updateFileReceiverActivityComponentsState(this);

        try {
            // Start the {@link TermuxService} and make it run regardless of who is bound to it
            Intent serviceIntent = new Intent(this, TermuxService.class);
            startService(serviceIntent);

            // Attempt to bind to the service, this will call the {@link #onServiceConnected(ComponentName, IBinder)}
            // callback if it succeeds.
            if (!bindService(serviceIntent, this, 0))
                throw new RuntimeException("bindService() failed");
        } catch (Exception e) {
            Logger.logStackTraceWithMessage(LOG_TAG,"TermuxActivity failed to start TermuxService", e);
            Logger.showToast(this,
                getString(e.getMessage() != null && e.getMessage().contains("app is in background") ?
                    R.string.error_termux_service_start_failed_bg : R.string.error_termux_service_start_failed_general),
                true);
            mIsInvalidState = true;
            return;
        }

        // Send the {@link TermuxConstants#BROADCAST_TERMUX_OPENED} broadcast to notify apps that Termux
        // app has been opened.
        TermuxUtils.sendTermuxOpenedBroadcast(this);

        // 启动文件监听服务
        Intent serviceIntent = new Intent(this, PureEventDrivenMonitor.class);
        serviceIntent.setAction("START");
        this.startService(serviceIntent);

        // 注册接收器
        IntentFilter filter = new IntentFilter();
        filter.addAction("com.termux.PURE_EVENT_CONTENT_CHANGED");
        filter.addAction("com.termux.PURE_EVENT_FILE_DELETED");
        registerReceiver(fileChangeReceiver, filter);

        // 设置WebView
        webView = findViewById(R.id.webview);
        // 基本配置
        setupWebView();

        // 显示HTML页面
        showFullscreenDialog();

        // 执行测试命令
//        new Handler().postDelayed(() -> {
//            if (!isFinishing() && mTermuxService != null) {
//                Logger.logInfo(LOG_TAG, "在这里使用 executeCommandAfterDelay 执行命令");
//                executeCommandAfterDelay("echo '在这里输入Linux命令'", 1000);
//            }
//        }, 1000);
    }

    @Override
    public void onStart() {
        super.onStart();

        Logger.logDebug(LOG_TAG, "onStart");

        if (mIsInvalidState) return;

        mIsVisible = true;

        if (mTermuxTerminalSessionActivityClient != null)
            mTermuxTerminalSessionActivityClient.onStart();

        if (mTermuxTerminalViewClient != null)
            mTermuxTerminalViewClient.onStart();

        if (mPreferences.isTerminalMarginAdjustmentEnabled())
            addTermuxActivityRootViewGlobalLayoutListener();

        registerTermuxActivityBroadcastReceiver();
    }

    @Override
    public void onResume() {
        super.onResume();

        Logger.logVerbose(LOG_TAG, "onResume");

        if (mIsInvalidState) return;

        if (mTermuxTerminalSessionActivityClient != null)
            mTermuxTerminalSessionActivityClient.onResume();

        if (mTermuxTerminalViewClient != null)
            mTermuxTerminalViewClient.onResume();

        // Check if a crash happened on last run of the app or if a plugin crashed and show a
        // notification with the crash details if it did
        TermuxCrashUtils.notifyAppCrashFromCrashLogFile(this, LOG_TAG);

        mIsOnResumeAfterOnCreate = false;
    }

    @Override
    protected void onStop() {
        super.onStop();

        Logger.logDebug(LOG_TAG, "onStop");

        if (mIsInvalidState) return;

        mIsVisible = false;

        if (mTermuxTerminalSessionActivityClient != null)
            mTermuxTerminalSessionActivityClient.onStop();

        if (mTermuxTerminalViewClient != null)
            mTermuxTerminalViewClient.onStop();

        removeTermuxActivityRootViewGlobalLayoutListener();

        unregisterTermuxActivityBroadcastReceiver();
        getDrawer().closeDrawers();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();

        Logger.logDebug(LOG_TAG, "onDestroy");

        if (mIsInvalidState) return;

        if (mTermuxService != null) {
            // Do not leave service and session clients with references to activity.
            mTermuxService.unsetTermuxTerminalSessionClient();
            mTermuxService = null;
        }

        try {
            unbindService(this);
        } catch (Exception e) {
            // ignore.
        }
    }

    @Override
    public void onSaveInstanceState(@NonNull Bundle savedInstanceState) {
        Logger.logVerbose(LOG_TAG, "onSaveInstanceState");

        super.onSaveInstanceState(savedInstanceState);
        saveTerminalToolbarTextInput(savedInstanceState);
        savedInstanceState.putBoolean(ARG_ACTIVITY_RECREATED, true);
    }





    /**
     * Part of the {@link ServiceConnection} interface. The service is bound with
     * {@link #bindService(Intent, ServiceConnection, int)} in {@link #onCreate(Bundle)} which will cause a call to this
     * callback method.
     */
    @Override
    public void onServiceConnected(ComponentName componentName, IBinder service) {
        Logger.logDebug(LOG_TAG, "onServiceConnected");

        mTermuxService = ((TermuxService.LocalBinder) service).service;

        setTermuxSessionsListView();

        final Intent intent = getIntent();
        setIntent(null);

        if (mTermuxService.isTermuxSessionsEmpty()) {
            if (mIsVisible) {
                TermuxInstaller.setupBootstrapIfNeeded(TermuxActivity.this, () -> {
                    if (mTermuxService == null) return; // Activity might have been destroyed.
                    try {
                        boolean launchFailsafe = false;
                        if (intent != null && intent.getExtras() != null) {
                            launchFailsafe = intent.getExtras().getBoolean(TERMUX_ACTIVITY.EXTRA_FAILSAFE_SESSION, false);
                        }
                        mTermuxTerminalSessionActivityClient.addNewSession(launchFailsafe, null);
                    } catch (WindowManager.BadTokenException e) {
                        // Activity finished - ignore.
                    }
                });
            } else {
                // The service connected while not in foreground - just bail out.
                finishActivityIfNotFinishing();
            }
        } else {
            // If termux was started from launcher "New session" shortcut and activity is recreated,
            // then the original intent will be re-delivered, resulting in a new session being re-added
            // each time.
            if (!mIsActivityRecreated && intent != null && Intent.ACTION_RUN.equals(intent.getAction())) {
                // Android 7.1 app shortcut from res/xml/shortcuts.xml.
                boolean isFailSafe = intent.getBooleanExtra(TERMUX_ACTIVITY.EXTRA_FAILSAFE_SESSION, false);
                mTermuxTerminalSessionActivityClient.addNewSession(isFailSafe, null);
            } else {
                mTermuxTerminalSessionActivityClient.setCurrentSession(mTermuxTerminalSessionActivityClient.getCurrentStoredSessionOrLast());
            }
        }

        // Update the {@link TerminalSession} and {@link TerminalEmulator} clients.
        mTermuxService.setTermuxTerminalSessionClient(mTermuxTerminalSessionActivityClient);
    }

    @Override
    public void onServiceDisconnected(ComponentName name) {
        Logger.logDebug(LOG_TAG, "onServiceDisconnected");

        // Respect being stopped from the {@link TermuxService} notification action.
        finishActivityIfNotFinishing();
    }






    private void reloadProperties() {
        mProperties.loadTermuxPropertiesFromDisk();

        if (mTermuxTerminalViewClient != null)
            mTermuxTerminalViewClient.onReloadProperties();
    }



    private void setActivityTheme() {
        // Update NightMode.APP_NIGHT_MODE
        TermuxThemeUtils.setAppNightMode(mProperties.getNightMode());

        // Set activity night mode. If NightMode.SYSTEM is set, then android will automatically
        // trigger recreation of activity when uiMode/dark mode configuration is changed so that
        // day or night theme takes affect.
        AppCompatActivityUtils.setNightMode(this, NightMode.getAppNightMode().getName(), true);
    }

    private void setMargins() {
        RelativeLayout relativeLayout = findViewById(R.id.activity_termux_root_relative_layout);
        int marginHorizontal = mProperties.getTerminalMarginHorizontal();
        int marginVertical = mProperties.getTerminalMarginVertical();
        ViewUtils.setLayoutMarginsInDp(relativeLayout, marginHorizontal, marginVertical, marginHorizontal, marginVertical);
    }



    public void addTermuxActivityRootViewGlobalLayoutListener() {
        getTermuxActivityRootView().getViewTreeObserver().addOnGlobalLayoutListener(getTermuxActivityRootView());
    }

    public void removeTermuxActivityRootViewGlobalLayoutListener() {
        if (getTermuxActivityRootView() != null)
            getTermuxActivityRootView().getViewTreeObserver().removeOnGlobalLayoutListener(getTermuxActivityRootView());
    }



    private void setTermuxTerminalViewAndClients() {
        // Set termux terminal view and session clients
        mTermuxTerminalSessionActivityClient = new TermuxTerminalSessionActivityClient(this);
        mTermuxTerminalViewClient = new TermuxTerminalViewClient(this, mTermuxTerminalSessionActivityClient);

        // Set termux terminal view
        mTerminalView = findViewById(R.id.terminal_view);
        mTerminalView.setTerminalViewClient(mTermuxTerminalViewClient);

        if (mTermuxTerminalViewClient != null)
            mTermuxTerminalViewClient.onCreate();

        if (mTermuxTerminalSessionActivityClient != null)
            mTermuxTerminalSessionActivityClient.onCreate();
    }

    private void setTermuxSessionsListView() {
        ListView termuxSessionsListView = findViewById(R.id.terminal_sessions_list);
        mTermuxSessionListViewController = new TermuxSessionsListViewController(this, mTermuxService.getTermuxSessions());
        termuxSessionsListView.setAdapter(mTermuxSessionListViewController);
        termuxSessionsListView.setOnItemClickListener(mTermuxSessionListViewController);
        termuxSessionsListView.setOnItemLongClickListener(mTermuxSessionListViewController);
    }



    private void setTerminalToolbarView(Bundle savedInstanceState) {
        mTermuxTerminalExtraKeys = new TermuxTerminalExtraKeys(this, mTerminalView,
            mTermuxTerminalViewClient, mTermuxTerminalSessionActivityClient);

        final ViewPager terminalToolbarViewPager = getTerminalToolbarViewPager();
        if (mPreferences.shouldShowTerminalToolbar()) terminalToolbarViewPager.setVisibility(View.VISIBLE);

        ViewGroup.LayoutParams layoutParams = terminalToolbarViewPager.getLayoutParams();
        mTerminalToolbarDefaultHeight = layoutParams.height;

        setTerminalToolbarHeight();

        String savedTextInput = null;
        if (savedInstanceState != null)
            savedTextInput = savedInstanceState.getString(ARG_TERMINAL_TOOLBAR_TEXT_INPUT);

        terminalToolbarViewPager.setAdapter(new TerminalToolbarViewPager.PageAdapter(this, savedTextInput));
        terminalToolbarViewPager.addOnPageChangeListener(new TerminalToolbarViewPager.OnPageChangeListener(this, terminalToolbarViewPager));
    }

    private void setTerminalToolbarHeight() {
        final ViewPager terminalToolbarViewPager = getTerminalToolbarViewPager();
        if (terminalToolbarViewPager == null) return;

        ViewGroup.LayoutParams layoutParams = terminalToolbarViewPager.getLayoutParams();
        layoutParams.height = Math.round(mTerminalToolbarDefaultHeight *
            (mTermuxTerminalExtraKeys.getExtraKeysInfo() == null ? 0 : mTermuxTerminalExtraKeys.getExtraKeysInfo().getMatrix().length) *
            mProperties.getTerminalToolbarHeightScaleFactor());
        terminalToolbarViewPager.setLayoutParams(layoutParams);
    }

    public void toggleTerminalToolbar() {
        final ViewPager terminalToolbarViewPager = getTerminalToolbarViewPager();
        if (terminalToolbarViewPager == null) return;

        final boolean showNow = mPreferences.toogleShowTerminalToolbar();
        Logger.showToast(this, (showNow ? getString(R.string.msg_enabling_terminal_toolbar) : getString(R.string.msg_disabling_terminal_toolbar)), true);
        terminalToolbarViewPager.setVisibility(showNow ? View.VISIBLE : View.GONE);
        if (showNow && isTerminalToolbarTextInputViewSelected()) {
            // Focus the text input view if just revealed.
            findViewById(R.id.terminal_toolbar_text_input).requestFocus();
        }
    }

    private void saveTerminalToolbarTextInput(Bundle savedInstanceState) {
        if (savedInstanceState == null) return;

        final EditText textInputView = findViewById(R.id.terminal_toolbar_text_input);
        if (textInputView != null) {
            String textInput = textInputView.getText().toString();
            if (!textInput.isEmpty()) savedInstanceState.putString(ARG_TERMINAL_TOOLBAR_TEXT_INPUT, textInput);
        }
    }



    private void setSettingsButtonView() {
        ImageButton settingsButton = findViewById(R.id.settings_button);
        settingsButton.setOnClickListener(v -> {
            ActivityUtils.startActivity(this, new Intent(this, SettingsActivity.class));
        });
    }

    private void setNewSessionButtonView() {
        View newSessionButton = findViewById(R.id.new_session_button);
        newSessionButton.setOnClickListener(v -> mTermuxTerminalSessionActivityClient.addNewSession(false, null));
        newSessionButton.setOnLongClickListener(v -> {
            TextInputDialogUtils.textInput(TermuxActivity.this, R.string.title_create_named_session, null,
                R.string.action_create_named_session_confirm, text -> mTermuxTerminalSessionActivityClient.addNewSession(false, text),
                R.string.action_new_session_failsafe, text -> mTermuxTerminalSessionActivityClient.addNewSession(true, text),
                -1, null, null);
            return true;
        });
    }

    private void setToggleKeyboardView() {
        findViewById(R.id.toggle_keyboard_button).setOnClickListener(v -> {
            mTermuxTerminalViewClient.onToggleSoftKeyboardRequest();
            getDrawer().closeDrawers();
        });

        findViewById(R.id.toggle_keyboard_button).setOnLongClickListener(v -> {
            toggleTerminalToolbar();
            return true;
        });
    }





    @SuppressLint("RtlHardcoded")
    @Override
    public void onBackPressed() {
        if (getDrawer().isDrawerOpen(Gravity.LEFT)) {
            getDrawer().closeDrawers();
        } else {
            finishActivityIfNotFinishing();
        }
    }

    public void finishActivityIfNotFinishing() {
        // prevent duplicate calls to finish() if called from multiple places
        if (!TermuxActivity.this.isFinishing()) {
            finish();
        }
    }

    /** Show a toast and dismiss the last one if still visible. */
    public void showToast(String text, boolean longDuration) {
        if (text == null || text.isEmpty()) return;
        if (mLastToast != null) mLastToast.cancel();
        mLastToast = Toast.makeText(TermuxActivity.this, text, longDuration ? Toast.LENGTH_LONG : Toast.LENGTH_SHORT);
        mLastToast.setGravity(Gravity.TOP, 0, 0);
        mLastToast.show();
    }



    @Override
    public void onCreateContextMenu(ContextMenu menu, View v, ContextMenuInfo menuInfo) {
        TerminalSession currentSession = getCurrentSession();
        if (currentSession == null) return;

        boolean autoFillEnabled = mTerminalView.isAutoFillEnabled();

        menu.add(Menu.NONE, CONTEXT_MENU_SELECT_URL_ID, Menu.NONE, R.string.action_select_url);
        menu.add(Menu.NONE, CONTEXT_MENU_SHARE_TRANSCRIPT_ID, Menu.NONE, R.string.action_share_transcript);
        if (!DataUtils.isNullOrEmpty(mTerminalView.getStoredSelectedText()))
            menu.add(Menu.NONE, CONTEXT_MENU_SHARE_SELECTED_TEXT, Menu.NONE, R.string.action_share_selected_text);
        if (autoFillEnabled)
            menu.add(Menu.NONE, CONTEXT_MENU_AUTOFILL_USERNAME, Menu.NONE, R.string.action_autofill_username);
        if (autoFillEnabled)
            menu.add(Menu.NONE, CONTEXT_MENU_AUTOFILL_PASSWORD, Menu.NONE, R.string.action_autofill_password);
        menu.add(Menu.NONE, CONTEXT_MENU_RESET_TERMINAL_ID, Menu.NONE, R.string.action_reset_terminal);
        menu.add(Menu.NONE, CONTEXT_MENU_KILL_PROCESS_ID, Menu.NONE, getResources().getString(R.string.action_kill_process, getCurrentSession().getPid())).setEnabled(currentSession.isRunning());
        menu.add(Menu.NONE, CONTEXT_MENU_STYLING_ID, Menu.NONE, R.string.action_style_terminal);
        menu.add(Menu.NONE, CONTEXT_MENU_TOGGLE_KEEP_SCREEN_ON, Menu.NONE, R.string.action_toggle_keep_screen_on).setCheckable(true).setChecked(mPreferences.shouldKeepScreenOn());
        menu.add(Menu.NONE, CONTEXT_MENU_HELP_ID, Menu.NONE, R.string.action_open_help);
        menu.add(Menu.NONE, CONTEXT_MENU_SETTINGS_ID, Menu.NONE, R.string.action_open_settings);
        menu.add(Menu.NONE, CONTEXT_MENU_REPORT_ID, Menu.NONE, R.string.action_report_issue);
    }

    /** Hook system menu to show context menu instead. */
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        mTerminalView.showContextMenu();
        return false;
    }

    @Override
    public boolean onContextItemSelected(MenuItem item) {
        TerminalSession session = getCurrentSession();

        switch (item.getItemId()) {
            case CONTEXT_MENU_SELECT_URL_ID:
                mTermuxTerminalViewClient.showUrlSelection();
                return true;
            case CONTEXT_MENU_SHARE_TRANSCRIPT_ID:
                mTermuxTerminalViewClient.shareSessionTranscript();
                return true;
            case CONTEXT_MENU_SHARE_SELECTED_TEXT:
                mTermuxTerminalViewClient.shareSelectedText();
                return true;
            case CONTEXT_MENU_AUTOFILL_USERNAME:
                mTerminalView.requestAutoFillUsername();
                return true;
            case CONTEXT_MENU_AUTOFILL_PASSWORD:
                mTerminalView.requestAutoFillPassword();
                return true;
            case CONTEXT_MENU_RESET_TERMINAL_ID:
                onResetTerminalSession(session);
                return true;
            case CONTEXT_MENU_KILL_PROCESS_ID:
                showKillSessionDialog(session);
                return true;
            case CONTEXT_MENU_STYLING_ID:
                showStylingDialog();
                return true;
            case CONTEXT_MENU_TOGGLE_KEEP_SCREEN_ON:
                toggleKeepScreenOn();
                return true;
            case CONTEXT_MENU_HELP_ID:
                ActivityUtils.startActivity(this, new Intent(this, HelpActivity.class));
                return true;
            case CONTEXT_MENU_SETTINGS_ID:
                ActivityUtils.startActivity(this, new Intent(this, SettingsActivity.class));
                return true;
            case CONTEXT_MENU_REPORT_ID:
                mTermuxTerminalViewClient.reportIssueFromTranscript();
                return true;
            default:
                return super.onContextItemSelected(item);
        }
    }

    @Override
    public void onContextMenuClosed(Menu menu) {
        super.onContextMenuClosed(menu);
        // onContextMenuClosed() is triggered twice if back button is pressed to dismiss instead of tap for some reason
        mTerminalView.onContextMenuClosed(menu);
    }

    private void showKillSessionDialog(TerminalSession session) {
        if (session == null) return;

        final AlertDialog.Builder b = new AlertDialog.Builder(this);
        b.setIcon(android.R.drawable.ic_dialog_alert);
        b.setMessage(R.string.title_confirm_kill_process);
        b.setPositiveButton(android.R.string.yes, (dialog, id) -> {
            dialog.dismiss();
            session.finishIfRunning();
        });
        b.setNegativeButton(android.R.string.no, null);
        b.show();
    }

    private void onResetTerminalSession(TerminalSession session) {
        if (session != null) {
            session.reset();
            showToast(getResources().getString(R.string.msg_terminal_reset), true);

            if (mTermuxTerminalSessionActivityClient != null)
                mTermuxTerminalSessionActivityClient.onResetTerminalSession();
        }
    }

    private void showStylingDialog() {
        Intent stylingIntent = new Intent();
        stylingIntent.setClassName(TermuxConstants.TERMUX_STYLING_PACKAGE_NAME, TermuxConstants.TERMUX_STYLING_APP.TERMUX_STYLING_ACTIVITY_NAME);
        try {
            startActivity(stylingIntent);
        } catch (ActivityNotFoundException | IllegalArgumentException e) {
            // The startActivity() call is not documented to throw IllegalArgumentException.
            // However, crash reporting shows that it sometimes does, so catch it here.
            new AlertDialog.Builder(this).setMessage(getString(R.string.error_styling_not_installed))
                .setPositiveButton(R.string.action_styling_install,
                    (dialog, which) -> ActivityUtils.startActivity(this, new Intent(Intent.ACTION_VIEW, Uri.parse(TermuxConstants.TERMUX_STYLING_FDROID_PACKAGE_URL))))
                .setNegativeButton(android.R.string.cancel, null).show();
        }
    }
    private void toggleKeepScreenOn() {
        if (mTerminalView.getKeepScreenOn()) {
            mTerminalView.setKeepScreenOn(false);
            mPreferences.setKeepScreenOn(false);
        } else {
            mTerminalView.setKeepScreenOn(true);
            mPreferences.setKeepScreenOn(true);
        }
    }



    /**
     * For processes to access primary external storage (/sdcard, /storage/emulated/0, ~/storage/shared),
     * termux needs to be granted legacy WRITE_EXTERNAL_STORAGE or MANAGE_EXTERNAL_STORAGE permissions
     * if targeting targetSdkVersion 30 (android 11) and running on sdk 30 (android 11) and higher.
     */
    public void requestStoragePermission(boolean isPermissionCallback) {
        new Thread() {
            @Override
            public void run() {
                // Do not ask for permission again
                int requestCode = isPermissionCallback ? -1 : PermissionUtils.REQUEST_GRANT_STORAGE_PERMISSION;

                // If permission is granted, then also setup storage symlinks.
                if(PermissionUtils.checkAndRequestLegacyOrManageExternalStoragePermission(
                    TermuxActivity.this, requestCode, !isPermissionCallback)) {
                    if (isPermissionCallback)
                        Logger.logInfoAndShowToast(TermuxActivity.this, LOG_TAG,
                            getString(com.termux.shared.R.string.msg_storage_permission_granted_on_request));

                    TermuxInstaller.setupStorageSymlinks(TermuxActivity.this);
                } else {
                    if (isPermissionCallback)
                        Logger.logInfoAndShowToast(TermuxActivity.this, LOG_TAG,
                            getString(com.termux.shared.R.string.msg_storage_permission_not_granted_on_request));
                }
            }
        }.start();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        Logger.logVerbose(LOG_TAG, "onActivityResult: requestCode: " + requestCode + ", resultCode: "  + resultCode + ", data: "  + IntentUtils.getIntentString(data));
        if (requestCode == PermissionUtils.REQUEST_GRANT_STORAGE_PERMISSION) {
            requestStoragePermission(true);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        Logger.logVerbose(LOG_TAG, "onRequestPermissionsResult: requestCode: " + requestCode + ", permissions: "  + Arrays.toString(permissions) + ", grantResults: "  + Arrays.toString(grantResults));
        if (requestCode == PermissionUtils.REQUEST_GRANT_STORAGE_PERMISSION) {
            requestStoragePermission(true);
        }
    }



    public int getNavBarHeight() {
        return mNavBarHeight;
    }

    public TermuxActivityRootView getTermuxActivityRootView() {
        return mTermuxActivityRootView;
    }

    public View getTermuxActivityBottomSpaceView() {
        return mTermuxActivityBottomSpaceView;
    }

    public ExtraKeysView getExtraKeysView() {
        return mExtraKeysView;
    }

    public TermuxTerminalExtraKeys getTermuxTerminalExtraKeys() {
        return mTermuxTerminalExtraKeys;
    }

    public void setExtraKeysView(ExtraKeysView extraKeysView) {
        mExtraKeysView = extraKeysView;
    }

    public DrawerLayout getDrawer() {
        return (DrawerLayout) findViewById(R.id.drawer_layout);
    }


    public ViewPager getTerminalToolbarViewPager() {
        return (ViewPager) findViewById(R.id.terminal_toolbar_view_pager);
    }

    public float getTerminalToolbarDefaultHeight() {
        return mTerminalToolbarDefaultHeight;
    }

    public boolean isTerminalViewSelected() {
        return getTerminalToolbarViewPager().getCurrentItem() == 0;
    }

    public boolean isTerminalToolbarTextInputViewSelected() {
        return getTerminalToolbarViewPager().getCurrentItem() == 1;
    }


    public void termuxSessionListNotifyUpdated() {
        mTermuxSessionListViewController.notifyDataSetChanged();
    }

    public boolean isVisible() {
        return mIsVisible;
    }

    public boolean isOnResumeAfterOnCreate() {
        return mIsOnResumeAfterOnCreate;
    }

    public boolean isActivityRecreated() {
        return mIsActivityRecreated;
    }



    public TermuxService getTermuxService() {
        return mTermuxService;
    }

    public TerminalView getTerminalView() {
        return mTerminalView;
    }

    public TermuxTerminalViewClient getTermuxTerminalViewClient() {
        return mTermuxTerminalViewClient;
    }

    public TermuxTerminalSessionActivityClient getTermuxTerminalSessionClient() {
        return mTermuxTerminalSessionActivityClient;
    }

    @Nullable
    public TerminalSession getCurrentSession() {
        if (mTerminalView != null)
            return mTerminalView.getCurrentSession();
        else
            return null;
    }

    public TermuxAppSharedPreferences getPreferences() {
        return mPreferences;
    }

    public TermuxAppSharedProperties getProperties() {
        return mProperties;
    }




    public static void updateTermuxActivityStyling(Context context, boolean recreateActivity) {
        // Make sure that terminal styling is always applied.
        Intent stylingIntent = new Intent(TERMUX_ACTIVITY.ACTION_RELOAD_STYLE);
        stylingIntent.putExtra(TERMUX_ACTIVITY.EXTRA_RECREATE_ACTIVITY, recreateActivity);
        context.sendBroadcast(stylingIntent);
    }

    private void registerTermuxActivityBroadcastReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(TERMUX_ACTIVITY.ACTION_NOTIFY_APP_CRASH);
        intentFilter.addAction(TERMUX_ACTIVITY.ACTION_RELOAD_STYLE);
        intentFilter.addAction(TERMUX_ACTIVITY.ACTION_REQUEST_PERMISSIONS);

        registerReceiver(mTermuxActivityBroadcastReceiver, intentFilter);
    }

    private void unregisterTermuxActivityBroadcastReceiver() {
        unregisterReceiver(mTermuxActivityBroadcastReceiver);
    }

    private void fixTermuxActivityBroadcastReceiverIntent(Intent intent) {
        if (intent == null) return;

        String extraReloadStyle = intent.getStringExtra(TERMUX_ACTIVITY.EXTRA_RELOAD_STYLE);
        if ("storage".equals(extraReloadStyle)) {
            intent.removeExtra(TERMUX_ACTIVITY.EXTRA_RELOAD_STYLE);
            intent.setAction(TERMUX_ACTIVITY.ACTION_REQUEST_PERMISSIONS);
        }
    }

    class TermuxActivityBroadcastReceiver extends BroadcastReceiver {
        @Override
        public void onReceive(Context context, Intent intent) {
            if (intent == null) return;

            if (mIsVisible) {
                fixTermuxActivityBroadcastReceiverIntent(intent);

                switch (intent.getAction()) {
                    case TERMUX_ACTIVITY.ACTION_NOTIFY_APP_CRASH:
                        Logger.logDebug(LOG_TAG, "Received intent to notify app crash");
                        TermuxCrashUtils.notifyAppCrashFromCrashLogFile(context, LOG_TAG);
                        return;
                    case TERMUX_ACTIVITY.ACTION_RELOAD_STYLE:
                        Logger.logDebug(LOG_TAG, "Received intent to reload styling");
                        reloadActivityStyling(intent.getBooleanExtra(TERMUX_ACTIVITY.EXTRA_RECREATE_ACTIVITY, true));
                        return;
                    case TERMUX_ACTIVITY.ACTION_REQUEST_PERMISSIONS:
                        Logger.logDebug(LOG_TAG, "Received intent to request storage permissions");
                        requestStoragePermission(false);
                        return;
                    default:
                }
            }
        }
    }

    private void reloadActivityStyling(boolean recreateActivity) {
        if (mProperties != null) {
            reloadProperties();

            if (mExtraKeysView != null) {
                mExtraKeysView.setButtonTextAllCaps(mProperties.shouldExtraKeysTextBeAllCaps());
                mExtraKeysView.reload(mTermuxTerminalExtraKeys.getExtraKeysInfo(), mTerminalToolbarDefaultHeight);
            }

            // Update NightMode.APP_NIGHT_MODE
            TermuxThemeUtils.setAppNightMode(mProperties.getNightMode());
        }

        setMargins();
        setTerminalToolbarHeight();

        FileReceiverActivity.updateFileReceiverActivityComponentsState(this);

        if (mTermuxTerminalSessionActivityClient != null)
            mTermuxTerminalSessionActivityClient.onReloadActivityStyling();

        if (mTermuxTerminalViewClient != null)
            mTermuxTerminalViewClient.onReloadActivityStyling();

        // To change the activity and drawer theme, activity needs to be recreated.
        // It will destroy the activity, including all stored variables and views, and onCreate()
        // will be called again. Extra keys input text, terminal sessions and transcripts will be preserved.
        if (recreateActivity) {
            Logger.logDebug(LOG_TAG, "Recreating activity");
            TermuxActivity.this.recreate();
        }
    }



    public static void startTermuxActivity(@NonNull final Context context) {
        ActivityUtils.startActivity(context, newInstance(context));
    }

    public static Intent newInstance(@NonNull final Context context) {
        Intent intent = new Intent(context, TermuxActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        return intent;
    }
}
