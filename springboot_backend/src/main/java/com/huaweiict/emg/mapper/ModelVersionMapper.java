package com.huaweiict.emg.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.huaweiict.emg.entity.ModelVersion;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

/**
 * 模型版本 Mapper
 */
@Mapper
public interface ModelVersionMapper extends BaseMapper<ModelVersion> {
    
    /**
     * 获取当前激活的模型版本
     */
    @Select("SELECT * FROM model_version WHERE is_active = TRUE LIMIT 1")
    ModelVersion selectActiveModel();
}
