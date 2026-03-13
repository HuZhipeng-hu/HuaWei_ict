package com.huaweiict.emg.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.huaweiict.emg.entity.TrainingTask;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;
import java.util.Map;

/**
 * 训练任务 Mapper
 */
@Mapper
public interface TrainingTaskMapper extends BaseMapper<TrainingTask> {
    
    /**
     * 获取训练任务摘要
     */
    @Select("SELECT * FROM v_training_task_summary ORDER BY created_time DESC LIMIT #{limit}")
    List<Map<String, Object>> getTaskSummary(int limit);
    
    /**
     * 获取正在运行的任务
     */
    @Select("SELECT * FROM training_task WHERE task_status = 'running'")
    List<TrainingTask> selectRunningTasks();
}
