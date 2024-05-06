import { defineStore } from 'pinia'
import {
  createTaskApi,
  getTasksApi,
  updateTaskApi,
  pauseTaskApi,
  resumeTaskApi
} from '@/api/federatedTask'
import { ElMessage } from 'element-plus'

interface TaskState {
  tasks: Array<{
    id: string
    name: string
    status: 'running' | 'paused' | 'completed'
    progress: number
  }>
}

export const useFederatedTaskStore = defineStore('federatedTask', {
  state: (): TaskState => ({
    tasks: []
  }),
  actions: {
    async createTask(taskData: any) {
      try {
        await createTaskApi(taskData)
        ElMessage.success('任务创建成功')
        this.tasks = await this.getTasks()
      } catch (error) {
        ElMessage.error('任务创建失败')
        console.error('创建任务失败:', error)
        throw error
      }
    },
    async getTasks() {
      try {
        const data = await getTasksApi()
        this.tasks = data || []
        return this.tasks
      } catch (error) {
        console.error('获取任务列表失败:', error)
        this.tasks = []
        return []
      }
    },
    async updateTask(taskData: any) {
      try {
        await updateTaskApi(taskData)
        ElMessage.success('任务更新成功')
        this.tasks = await this.getTasks()
      } catch (error) {
        ElMessage.error('任务更新失败')
        console.error('更新任务失败:', error)
        throw error
      }
    },
    async pauseTask(taskId: string) {
      try {
        await pauseTaskApi(taskId)
        ElMessage.info('任务已暂停')
        this.tasks = await this.getTasks()
      } catch (error) {
        ElMessage.error('暂停任务失败')
        console.error('暂停任务失败:', error)
        throw error
      }
    },
    async resumeTask(taskId: string) {
      try {
        await resumeTaskApi(taskId)
        ElMessage.success('任务已恢复')
        this.tasks = await this.getTasks()
      } catch (error) {
        ElMessage.error('恢复任务失败')
        console.error('恢复任务失败:', error)
        throw error
      }
    }
  }
})
