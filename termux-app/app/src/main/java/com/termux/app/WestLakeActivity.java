package com.termux.app;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import com.termux.shared.logger.Logger;
import com.termux.shared.termux.TermuxConstants;

import com.termux.R;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;

public class WestLakeActivity extends Activity {
    private EditText etCommandInput;
    private TextView tvOutput;
    private Button btnSwitchToTerminal;
    private Button btnExecuteCommand;

    private TermuxService termuxService;
    private Handler mainHandler;

    private static final String LOG_TAG = "WestLakeActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_westlake);

        initViews();
        setupClickListeners();
        mainHandler = new Handler(Looper.getMainLooper());

        // 绑定Termux服务
        bindTermuxService();

        // 复制 termux-login.sh 文件
        copyTermuxLoginScript();
    }

    /**
     * 复制 termux-login.sh 到 PREFIX/etc/ 目录
     */
    private void copyTermuxLoginScript() {
        new Thread(() -> {
            try {
                // 等待环境初始化完成
                Thread.sleep(1000);

                File etcDir = new File(TermuxConstants.TERMUX_PREFIX_DIR, "etc");
                File destFile = new File(etcDir, "termux-login.sh");

                // 如果目标文件已存在，跳过复制
//                if (destFile.exists()) {
//                    Logger.logVerbose(LOG_TAG, "termux-login.sh already exists, skipping copy");
//                    return;
//                }

                // 确保目标目录存在
                if (!etcDir.exists()) {
                    if (!etcDir.mkdirs()) {
                        Logger.logError(LOG_TAG, "Failed to create etc directory");
                        return;
                    }
                }

                // 从 assets 复制文件
                copyTermuxLoginScriptFromAssets(destFile);

            } catch (Exception e) {
                Logger.logError(LOG_TAG, "Failed to copy termux-login.sh: " + e.getMessage());
            }
        }).start();
    }

    /**
     * 从 assets 目录复制 termux-login.sh 文件
     */
    private void copyTermuxLoginScriptFromAssets(File destFile) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            try (InputStream in = getAssets().open("termux-login.sh");
                 OutputStream out = Files.newOutputStream(destFile.toPath())) {

                byte[] buffer = new byte[1024];
                int length;
                while ((length = in.read(buffer)) > 0) {
                    out.write(buffer, 0, length);
                }

                // 设置文件权限为可执行
                setFileExecutable(destFile);

                Logger.logVerbose(LOG_TAG, "Successfully copied termux-login.sh to " + destFile.getAbsolutePath());

            } catch (IOException e) {
                Logger.logError(LOG_TAG, "Error copying termux-login.sh from assets: " + e.getMessage());
            }
        }
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

    private void initViews() {
        etCommandInput = findViewById(R.id.et_command_input);
        tvOutput = findViewById(R.id.tv_output);
        btnSwitchToTerminal = findViewById(R.id.btn_switch_to_terminal);
        btnExecuteCommand = findViewById(R.id.btn_execute_command);

        // 设置TextView可滚动
        tvOutput.setMovementMethod(new ScrollingMovementMethod());
    }

    private void setupClickListeners() {
        // 切换到原终端界面
        btnSwitchToTerminal.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                switchToTerminalActivity();
            }
        });

        // 执行命令
        btnExecuteCommand.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
            }
        });
    }

    private void bindTermuxService() {
        // 这里需要连接到TermuxService
        // 实际实现需要参考Termux的Service绑定机制
        Intent serviceIntent = new Intent(this, TermuxService.class);
        startService(serviceIntent);
    }

    private void switchToTerminalActivity() {
        Intent intent = new Intent(this, TermuxActivity.class);
        intent.putExtra("FROM_CUSTOM_ACTIVITY", true);
        startActivity(intent);
        // finish(); // 关闭当前界面
    }

    private void executeCommand(String command) {
        if (command == null || command.trim().isEmpty()) {
            appendOutput("错误：命令不能为空");
            return;
        }

        appendOutput("执行命令: " + command);

        // 这里需要调用Termux的API来执行命令
        // 实际实现需要与TermuxSession交互
        try {
            // 伪代码 - 实际实现需要访问TermuxService
            executeCommandInTerminal(command);
        } catch (Exception e) {
            appendOutput("执行命令出错: " + e.getMessage());
        }
    }

    private void executeCommandInTerminal(String command) {
        // 这里需要实现与Termux终端的交互
        // 可能需要通过TermuxService或TermuxSession

        // 伪代码示例：
        /*
        if (termuxService != null) {
            TermuxSession currentSession = termuxService.getCurrentSession();
            if (currentSession != null) {
                currentSession.getTerminalIO().write(command + "\n");
            }
        }
        */
    }

    private void appendOutput(final String text) {
        mainHandler.post(new Runnable() {
            @SuppressLint("SetTextI18n")
            @Override
            public void run() {
                String currentText = tvOutput.getText().toString();
                tvOutput.setText(currentText + "\n" + text);

                // 自动滚动到底部
                final int scrollAmount = tvOutput.getLayout().getLineTop(tvOutput.getLineCount()) - tvOutput.getHeight();
                tvOutput.scrollTo(0, Math.max(scrollAmount, 0));
            }
        });
    }

    // 接收命令执行结果的方法
    public void onCommandOutput(String output) {
        appendOutput("输出: " + output);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // 清理资源
        if (mainHandler != null) {
            mainHandler.removeCallbacksAndMessages(null);
        }
    }
}
