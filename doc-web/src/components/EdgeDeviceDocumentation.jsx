import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { marked } from 'marked'
import styles from './EdgeDeviceDocumentation.module.css'

// EdgeDeviceDocumentation content mapping
const docs = {
  'guide': {
    title: 'Guide',
    content: `# Edge Device Documentation

## Documentation Index

- [Environmental installation](device/install) - Installing a runtime environment on mobile devices
- [Start the service](device/start) - System deployment and operations
- [Train the model](device/train) - Model training on edge devices
- [Chat](device/chat) - Local or cloud based model dialogue capability

## Download the application

1. [termux-app_apt-android-7-release_initialize.apk](termux-app_apt-android-7-release_initialize.apk)
2. [termux-app_apt-android-7-release_universal.apk](termux-app_apt-android-7-release_universal.apk)

Due to the inability of the modified Termux app to start the internal initialization program properly, we need to first install and initialize the Termux app, and then install the official Termux app

## Contact

For questions, please check the relevant documentation or submit an Issue.
`
  },
  'install': {
    title: 'Install',
    content: `# Edge Device Installation
Due to environmental limitations, the default system provided by \`Termux\` is not a real Linux environment, so it is necessary to install \`proot-distro\` and set up a subsystem such as \`Ubuntu\` or other available distributions to run.

## Installing the Application
A modified Termux application is provided, which will automatically transfer some convenient files into the environment and also offer visual training and conversation scenarios.

## Automatic Installation
Open the \`Termux\` application and execute \`./install.sh\`, then wait for completion.

Note 1: A proxy may need to be enabled as some dependency libraries may require access to \`GitHub\`.
Note 2: Due to network limitations, installation may fail. Check the log information. If automatic installation fails, switch to manual installation.

## Manual Installation
Follow the steps below to set up the environment manually:
\`\`\`
# 1. Update (note: enter Y when prompted to agree)
apt-get update

# 2. Install proot-distro
pkg install proot-distro

# 3. Install a subsystem (Ubuntu is recommended)
proot-distro install ubuntu

# 4. Log into the subsystem (Ubuntu)
proot-distro login ubuntu

# 5. Update the system (note: enter Y when prompted to agree)
apt-get update

# 6. Install Python
apt install python3
apt install python3-venv

# 7. Activate the virtual environment
python3 -m venv pytorch_env

# 8. Install PyTorch and related machine learning libraries
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 9. Install common data science libraries (optional)
pip3 install numpy pandas matplotlib scikit-learn jupyter
\`\`\`

Afterward, you can verify whether PyTorch is installed successfully:
\`\`\`
python3 -c \\"
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('✅ PyTorch installation successful!')
\\"
\`\`\`

Copy the example project:
\`\`\`
mkdir -p train_example_1
cp --update=none /data/data/com.termux/files/home/train_example_1/* ~/train_example_1/
\`\`\`

## Installing Ollama
\`Ollama\` is used to manage local and cloud models and supports conversations through a frontend interface. For more detailed information about \`Ollama\`, please visit the official website.

### Installation
\`\`\`
proot-distro login ubuntu
curl -fsSL https://ollama.com/install.sh | sh
\`\`\`
It is recommended to install it within the subsystem.

### Launching
\`\`\`
proot-distro login ubuntu
ollama serve
\`\`\`
Note: The application allows you to swipe from the left to open the drawer panel and select \`NEW SESSION\` to add a new terminal session. Start the \`Ollama\` service in this new session, as the \`Ollama\` service will occupy the input and may cause the frontend interface operations to freeze. Therefore, ensure the first session remains in its default state. If you perform other operations in it, exit the subsystem after completion, navigate back to \`~/\`, and reopen the frontend interface for operations.
<br/>
TODO: Automatically start a new \`SESSION\` for frontend command interactions to resolve this.

## Directory Guide
After launching \`Termux\`, you will see several preset directories by default:

\`\`\`
models
\`\`\`
Dynamically generated; may not exist by default. Models need to be synchronized in the conversation interface. The application will download models based on the model names from the server.

\`\`\`
train_example_1
\`\`\`
Preset \`mnist\` handwriting recognition training PyTorch code. Based on the number of preset training code examples, it will automatically increment, e.g., \`train_example_2\`, \`train_example_123\`.

\`\`\`
transmit
\`\`\`
Used to transfer JSON data from the terminal to the frontend. The application uses a file event mechanism to automatically forward data to the frontend.
Note: If the necessary \`Utils\` class and related settings are not used in the training code, the frontend will be unable to monitor it.

## Important Warnings

Caution: In the \`Android\` system, due to the intrusive nature of code development, do **not** use **\`Clear Data\`** to reset the system, as this may prevent the device from initializing correctly.

Note: On some mobile devices, initialization may fail for reasons currently under investigation.
    `
  },
  'start': {
    title: 'Start',
    content: `# Edge Device Start
## Interface Introduction

### Default Interface
By default, when the \`Termux\` application starts, it automatically opens the frontend interface, as shown below:

![image.png](https://api.apifox.com/api/v1/projects/7540101/resources/603851/image-preview)

Once the device starts federated learning, this area will display the connection status and show the currently connected device ID.

![image.png](https://api.apifox.com/api/v1/projects/7540101/resources/605492/image-preview)

### Settings Interface
Clicking \`Settings\` on the right side of \`Function Selection\` will open the settings page:

![image.png](https://api.apifox.com/api/v1/projects/7540101/resources/605494/image-preview)

Set the \`Ollama Host\` to manage local models. If not set or if the service is not running, local models will not be accessible.

Set the \`Server Host\` to connect to the remote main server.

Set the \`Server UserName\` and \`Server Password\` for authentication with the remote main server account (note: the password will not be saved; you need to re-enter it each time after exiting the \`APP\`).

### Terminal Interface
Click \`Open terminal\` to switch to the terminal interface.

![image.png](https://api.apifox.com/api/v1/projects/7540101/resources/604425/image-preview)

Click the button to open the \`SESSION\` drawer panel:
![image.png](https://api.apifox.com/api/v1/projects/7540101/resources/604464/image-preview)

Please keep the first \`SESSION\` for handling interactions with the frontend interface. Start services in the second and third \`SESSION\`s.

## Starting Services

Since \`Termux\` does not yet have automatic \`SESSION\` and command functionality, you need to manually create two new \`SESSION\`s and manually start the following two services:

### Ollama
Log into the subsystem (execute in the third \`SESSION\`):
\`\`\`
proot-distro login ubuntu
\`\`\`

Expose the \`OLLAMA_HOST\` port (optional):
\`\`\`
export OLLAMA_HOST=0.0.0.0:11434
\`\`\`

Start the \`Ollama\` service:
\`\`\`
ollama serve
\`\`\`

### Training Task

Log into the subsystem (execute in the second \`SESSION\`):
\`\`\`
proot-distro login ubuntu
\`\`\`

Activate the default Python virtual environment:
\`\`\`
source pytorch_env/bin/activate
\`\`\`

Navigate to the \`train_example_1\` directory:
\`\`\`
cd train_example_1
\`\`\`

Install dependencies:
\`\`\`
pip install -r requirements.txt
\`\`\`

Start the training task listener:
\`\`\`
python3 start_device.py
\`\`\`
Started via MQTT by default. The listener periodically sends heartbeats and will begin processing training tasks upon receiving them.`,
  },
  'train': {
    title: 'Training',
    content: `# Edge Device Training
## Utility Classes
Since data needs to be passed to the frontend to determine training progress and related information, utility classes are required to write necessary data into JSON files within the \`transmit\` directory.

### Updating Current Training Device Information

#### Converting Regional Server Data for Frontend Transmission

In the file \`trainer.py\`:

Ensure the following exists:
\`\`\`
self.current_round = 0
self.total_rounds = 0
\`\`\`

Set in the \`fit()\` function and update immediately after receiving parameters from the regional server:
\`\`\`
# Initialize
utils = Utils()

# Get data from device_info.json
device_info = utils.get_device_info()

# Set training count and current training index position
self.current_round = device_info['train']['index'] + 1
self.total_rounds = device_info['train']['round']

# Update device information
if self.total_rounds > 0:
    utils.update_device_info(train={
        'index': self.current_round,
        'round': self.total_rounds,
        'progress': (self.current_round / self.total_rounds) * 100
    })
else:
    utils.update_device_info(train={
        'index': self.current_round,
        'round': self.total_rounds,
    })
\`\`\`

#### Update During Each Model Evaluation
To ensure timely viewing of model training progress, add the following code inside the evaluation method:
\`\`\`
utils = Utils()
utils.update_device_info(
    train={
        'accuracy': accuracy,
        'loss': loss,
        'correct': correct,
        'total': total,
        'progress': (self.current_round / self.total_rounds) * 100
    }
)
\`\`\`

#### Update Device Status When MQTT Sends Heartbeat (Optional)
The frontend retains device status for 60 seconds by default. Once the heartbeat stops for longer than this period, the connection will be automatically disconnected.
\`\`\`
utils = Utils()
device_info = utils.get_device_info()
if device_info['device']['status'] != "online":
    utils.update_device_info(
        device={"status": "online"},
    )
else:
    utils.update_device_info(
        device={"timestamp": int(datetime.now().timestamp())},
    )
\`\`\`
If the \`MQTT\` channel closes, it should be set to offline:
\`\`\`
utils = Utils()
device_info = utils.get_device_info()
utils.update_device_info(
    device={
        "status": "offline",
        "timestamp": int(datetime.now().timestamp())
    },
)
\`\`\`

The example project provides \`HTTPClient\`. Please read the code in detail. You can also use this class directly to build custom training model code.

#### Regional Server Data Transmission Example
\`\`\`
{
  "action": "task_start",
  "task_id": 69,
  "task_name": "cnn10",
  "model_info": {
    "id": 1,
    "name": "minst",
    "description": "Handwriting Recognition"
  },
  "model_version": {
    "id": 1,
    "version": "0.0.1",
    "model_file": "models/1761630537818_none.pt",
    "description": "None",
    "accuracy": "None",
    "loss": "None",
    "metrics": {}
  },
  "rounds": 10,
  "aggregation_method": "fedavg",
  "device_info": {
    "device_id": "device_3884d708",
    "ip_address": "None",
    "device_context": {
      "status": "online",
      "timestamp": 1764581458.9900446,
      "current_task": "None"
    },
    "status": "online",
    "last_heartbeat": "2025-12-01T09:30:58.991858+00:00",
    "description": "None"
  },
  "flower_server": {
    "host": "127.0.0.1",
    "port": 8080,
    "server_id": "federated_server_69",
    "running": "True"
  }
}
\`\`\`

#### Frontend Data File Transmission Example
\`\`\`
{
    "region": {
        "id": ""
    },
    "task": {
        "id": "",
        "name": ""
    },
    "model": {
        "id": "",
        "name": "",
        "description": "",
        "file": ""
    },
    "train": {
        "index": 0,
        "round": 0,
        "aggregation_method": null,
        "progress": 0,
        "accuracy": 0,
        "loss": 0,
        "correct": 0,
        "total": 0
    },
    "device": {
        "id": "",
        "status": "offline",
        "timestamp": "",
        "description": ""
    },
    "flower_server": {
        "host": "127.0.0.1",
        "port": 8080,
        "server_id": "",
        "running": ""
    }
}
\`\`\`

## How to Transfer Files to the Device

1. Open the terminal.
2. Create a new \`SESSION\` (optional).
3. Enter \`termux-setup-storage\` and allow file access permissions.

Using the following command, you will see a list of files based on the phone's file manager:
\`\`\`
cd storage/shared
\`\`\`
The directory in the terminal environment is: \`~/storage/shared\`

You can then copy files into the terminal environment, for example:
\`\`\`
cp gemma-3-270m-it-UD-Q4_K_XL.gguf ~/
\`\`\`
After execution, use:
\`\`\`
cd ~/
\`\`\`
You will see the \`gguf\` file copied to the terminal.

## How to Transfer Terminal Files to the Subsystem

Since we have installed a subsystem in the terminal, we need to continue moving files so they can be copied into the subsystem.

First, log into the subsystem:
\`\`\`
proot-distro login ubuntu
\`\`\`
Navigate to the following path:
\`\`\`
cd /data/data/com.termux/files/home
\`\`\`
You will see all files from the terminal. You can now use the \`cp\` command to move directories or files to the root directory of the subsystem:
\`\`\`
cp -r train_example_1/ ~/
\`\`\`
Then return to the subsystem root directory:
\`\`\`
cd ~/
\`\`\`
Using \`ls\`, you can see that the \`train_example_1\` directory has been moved from the terminal to the subsystem.

## Starting Training

First, activate the Python virtual environment:
\`\`\`
source pytorch_env/bin/activate
\`\`\`
It is recommended to create independent virtual environments in different project directories to avoid interference between dependency libraries and ensure good compatibility.

Navigate to the example project directory:
\`\`\`
cd train_example_1
\`\`\`

Modify the \`.env\` file:
\`\`\`
nano .env
\`\`\`
Example content:
\`\`\`
DEVICE_ID=device_001
REGION_ID=7
MQTT_BROKER_HOST=192.168.1.2
CENTRAL_SERVER_URL=http://192.168.1.2:8085
\`\`\`

**DEVICE_ID** Setting
![image.png](https://api.apifox.com/api/v1/projects/7540101/resources/604961/image-preview)
Note: Since adding edge devices to the main server requires entering an \`IP\` address, you need to obtain it in advance on the mobile device using the following command:
\`\`\`
ifconfig
\`\`\`
You will find a local network \`IP\` address such as: \`192.168.1.8\`. Note: Ensure the mobile device and regional server are on the same network.

**REGION_ID** Setting
![image.png](https://api.apifox.com/api/v1/projects/7540101/resources/604960/image-preview)

**MQTT_BROKER_HOST** Setting
Refer to the \`IP\` address where the regional server is built.

**CENTRAL_SERVER_URL** Setting
Refer to the \`IP\` address and port where the regional server is built, and ensure the network connection is smooth.`,
  },
  'chat': {
    title: 'Chat',
    content: `# Edge Device Chat Models
The conversational model supports both local model conversations and cloud conversations. Local conversation histories can be uploaded to the cloud.

![image.png](https://api.apifox.com/api/v1/projects/7540101/resources/604964/image-preview)

## Syncing Models

You can set up models in the main server backend:

![image.png](https://api.apifox.com/api/v1/projects/7540101/resources/604966/image-preview)

On the mobile device, the configured models will be automatically displayed:
![image.png](https://api.apifox.com/api/v1/projects/7540101/resources/604965/image-preview)

Then click on a model, and select one of its versions from the pop-up list to sync:

![image.png](https://api.apifox.com/api/v1/projects/7540101/resources/604967/image-preview)

This will download the model to the mobile device and automatically deploy it using \`Ollama\` (Note: Since uploading models only supports \`.pt\` and \`.zip\` files, \`Ollama\` might not deploy them correctly).

## Conversational Model

After selecting a model, the conversation window will appear below:

![image.png](https://api.apifox.com/api/v1/projects/7540101/resources/604968/image-preview)

Once a conversation is completed, you can use the save option to upload the conversation history.`,
  }
}

export default function EdgeDeviceDocumentation() {
  const { docId } = useParams()
  const [currentDoc, setCurrentDoc] = useState('guide')
  const [htmlContent, setHtmlContent] = useState('')

  useEffect(() => {
    const docKey = docId || 'guide'
    setCurrentDoc(docKey)
    
    if (docs[docKey]) {
      // Configure marked options
      marked.setOptions({
        breaks: true,
        gfm: true,
        headerIds: true,
        mangle: false
      })
      
      const html = marked(docs[docKey].content)
      setHtmlContent(html)
    }
  }, [docId])

  const tocItems = [
    { id: 'guide', title: 'Guide', icon: '📚' },
    { id: 'install', title: 'Install', icon: '🏗️' },
    { id: 'start', title: 'Start', icon: '🚀' },
    { id: 'train', title: 'Training', icon: '💻' },
    { id: 'chat', title: 'Chat', icon: '🤖' },

  ]

  return (
    <div className={styles.documentation}>
      <aside className={styles.sidebar}>
        <div className={styles.sidebarHeader}>
          <h2>📖 Documentation</h2>
        </div>
        <nav className={styles.toc}>
          {tocItems.map(item => (
            <Link
              key={item.id}
              to={`/device/${item.id}`}
              className={`${styles.tocItem} ${currentDoc === item.id ? styles.active : ''}`}
            >
              <span className={styles.tocIcon}>{item.icon}</span>
              <span className={styles.tocTitle}>{item.title}</span>
            </Link>
          ))}
        </nav>
      </aside>
      
      <main className={styles.content}>
        <div className={styles.contentWrapper}>
          <article 
            className={styles.markdown}
            dangerouslySetInnerHTML={{ __html: htmlContent }}
          />
        </div>
      </main>
    </div>
  )
}
