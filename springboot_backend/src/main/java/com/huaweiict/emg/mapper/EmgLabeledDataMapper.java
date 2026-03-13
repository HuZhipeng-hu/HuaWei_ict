package com.huaweiict.emg.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.huaweiict.emg.entity.EmgLabeledData;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * EMG标注数据 Mapper
 */
@Mapper
public interface EmgLabeledDataMapper extends BaseMapper<EmgLabeledData> {
    
    /**
     * 统计各手势的样本数
     */
    @Select("SELECT gesture_label, COUNT(*) as count FROM emg_labeled_data " +
            "WHERE is_valid = TRUE GROUP BY gesture_label")
    List<Map<String, Object>> countByGesture();
    
    /**
     * 查询时间范围内的数据
     */
    @Select("SELECT * FROM emg_labeled_data " +
            "WHERE device_id = #{deviceId} " +
            "AND capture_time BETWEEN #{startTime} AND #{endTime} " +
            "ORDER BY capture_time")
    List<EmgLabeledData> selectByTimeRange(String deviceId, LocalDateTime startTime, LocalDateTime endTime);
    
    /**
     * 获取标注统计信息
     */
    @Select("SELECT * FROM v_annotation_statistics")
    List<Map<String, Object>> getAnnotationStatistics();
}
