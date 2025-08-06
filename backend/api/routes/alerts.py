"""
Alert and Notification API Routes
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from datetime import datetime, timedelta

from core.notifications.alert_manager import (
    alert_manager,
    AlertType,
    AlertPriority,
    NotificationChannel,
    AlertStatus,
    AlertConfig,
    ChannelConfig,
    AlertNotification,
    NotificationTemplate
)
from core.auth import get_current_user
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.post("/send")
async def send_alert(
    alert_type: AlertType,
    priority: AlertPriority,
    title: str,
    message: str,
    symbol: Optional[str] = None,
    agent_id: Optional[str] = None,
    portfolio_id: Optional[int] = None,
    data: Dict[str, Any] = Body(default={}),
    recipients: List[str] = Body(default=[]),
    current_user=Depends(get_current_user)
) -> Dict[str, str]:
    """
    Send a custom alert
    
    Args:
        alert_type: Type of alert
        priority: Alert priority
        title: Alert title
        message: Alert message
        symbol: Optional symbol
        agent_id: Optional agent ID
        portfolio_id: Optional portfolio ID
        data: Additional data
        recipients: List of recipients
        
    Returns:
        Alert ID
    """
    try:
        alert_id = await alert_manager.send_alert(
            alert_type=alert_type,
            priority=priority,
            title=title,
            message=message,
            symbol=symbol,
            agent_id=agent_id,
            portfolio_id=portfolio_id,
            data=data,
            recipients=recipients
        )
        
        if alert_id:
            return {
                "success": True,
                "alert_id": alert_id,
                "message": "Alert sent successfully"
            }
        else:
            return {
                "success": False,
                "message": "Alert not sent (disabled or filtered)"
            }
        
    except Exception as e:
        logger.error(f"Failed to send alert: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to send alert")


@router.get("/history")
async def get_alert_history(
    alert_type: Optional[AlertType] = None,
    priority: Optional[AlertPriority] = None,
    limit: int = Query(100, ge=1, le=1000),
    current_user=Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Get alert history
    
    Args:
        alert_type: Filter by alert type
        priority: Filter by priority
        limit: Maximum number of alerts to return
        
    Returns:
        List of alerts
    """
    try:
        alerts = await alert_manager.get_alert_history(
            alert_type=alert_type,
            priority=priority,
            limit=limit
        )
        
        return [
            {
                "alert_id": alert.alert_id,
                "type": alert.alert_type.value,
                "priority": alert.priority.value,
                "title": alert.title,
                "message": alert.message,
                "symbol": alert.symbol,
                "status": alert.status.value,
                "created_at": alert.created_at,
                "sent_at": alert.sent_at,
                "acknowledged_at": alert.acknowledged_at,
                "channels": [c.value for c in alert.channels],
                "data": alert.data
            }
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error(f"Failed to get alert history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alert history")


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user=Depends(get_current_user)
) -> Dict[str, str]:
    """
    Acknowledge an alert
    
    Args:
        alert_id: Alert ID to acknowledge
        
    Returns:
        Confirmation message
    """
    try:
        success = await alert_manager.acknowledge_alert(alert_id)
        
        if success:
            return {"message": "Alert acknowledged successfully"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")


@router.get("/config")
async def get_alert_configs(
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get all alert configurations
    
    Returns:
        Alert configurations
    """
    configs = {}
    
    for alert_type, config in alert_manager.alert_configs.items():
        configs[alert_type.value] = {
            "enabled": config.enabled,
            "channels": [c.value for c in config.channels],
            "min_priority": config.min_priority.value,
            "cooldown_minutes": config.cooldown_minutes,
            "aggregate": config.aggregate,
            "min_confidence": config.min_confidence,
            "signal_types": config.signal_types,
            "active_hours": config.active_hours,
            "active_days": config.active_days
        }
    
    return configs


@router.put("/config/{alert_type}")
async def update_alert_config(
    alert_type: AlertType,
    config: AlertConfig,
    current_user=Depends(get_current_user)
) -> Dict[str, str]:
    """
    Update alert configuration
    
    Args:
        alert_type: Alert type to configure
        config: New configuration
        
    Returns:
        Confirmation message
    """
    try:
        alert_manager.set_alert_config(alert_type, config)
        return {"message": f"Configuration updated for {alert_type.value}"}
        
    except Exception as e:
        logger.error(f"Failed to update alert config: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update configuration")


@router.get("/channels")
async def get_channel_configs(
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get all channel configurations
    
    Returns:
        Channel configurations
    """
    configs = {}
    
    for channel, config in alert_manager.channel_configs.items():
        channel_data = {
            "enabled": config.enabled
        }
        
        # Add channel-specific fields (excluding sensitive data)
        if channel == NotificationChannel.EMAIL:
            channel_data.update({
                "smtp_host": config.smtp_host,
                "smtp_port": config.smtp_port,
                "from_email": config.from_email,
                "configured": bool(config.smtp_host and config.from_email)
            })
        elif channel == NotificationChannel.WEBHOOK:
            channel_data.update({
                "webhook_url": config.webhook_url,
                "configured": bool(config.webhook_url)
            })
        elif channel == NotificationChannel.TELEGRAM:
            channel_data.update({
                "chat_id": config.telegram_chat_id,
                "configured": bool(config.telegram_token and config.telegram_chat_id)
            })
        elif channel == NotificationChannel.SLACK:
            channel_data.update({
                "channel": config.slack_channel,
                "configured": bool(config.slack_webhook)
            })
        elif channel == NotificationChannel.DISCORD:
            channel_data.update({
                "configured": bool(config.discord_webhook)
            })
        
        configs[channel.value] = channel_data
    
    return configs


@router.put("/channels/{channel}")
async def update_channel_config(
    channel: NotificationChannel,
    config: ChannelConfig,
    current_user=Depends(get_current_user)
) -> Dict[str, str]:
    """
    Update channel configuration
    
    Args:
        channel: Channel to configure
        config: New configuration
        
    Returns:
        Confirmation message
    """
    try:
        alert_manager.set_channel_config(channel, config)
        return {"message": f"Configuration updated for {channel.value}"}
        
    except Exception as e:
        logger.error(f"Failed to update channel config: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update configuration")


@router.post("/test/{channel}")
async def test_channel(
    channel: NotificationChannel,
    test_message: str = "This is a test alert from GoldenSignals AI",
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Test a notification channel
    
    Args:
        channel: Channel to test
        test_message: Test message to send
        
    Returns:
        Test result
    """
    try:
        # Create test alert
        test_alert = AlertNotification(
            alert_type=AlertType.SYSTEM,
            priority=AlertPriority.LOW,
            title="Test Alert",
            message=test_message,
            channels=[channel],
            data={"test": True}
        )
        
        # Try to send
        try:
            await alert_manager._send_to_channel(test_alert, channel)
            return {
                "success": True,
                "message": f"Test alert sent via {channel.value}"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to send via {channel.value}",
                "error": str(e)
            }
        
    except Exception as e:
        logger.error(f"Failed to test channel: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to test channel")


@router.get("/templates")
async def get_templates(
    current_user=Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Get notification templates
    
    Returns:
        List of templates
    """
    return [
        {
            "template_id": template.template_id,
            "name": template.name,
            "alert_type": template.alert_type.value,
            "channel": template.channel.value,
            "format": template.format,
            "variables": template.variables
        }
        for template in alert_manager.templates.values()
    ]


@router.post("/templates")
async def create_template(
    template: NotificationTemplate,
    current_user=Depends(get_current_user)
) -> Dict[str, str]:
    """
    Create or update a notification template
    
    Args:
        template: Template to create/update
        
    Returns:
        Confirmation message
    """
    try:
        alert_manager.templates[template.template_id] = template
        return {
            "message": f"Template {template.name} created/updated",
            "template_id": template.template_id
        }
        
    except Exception as e:
        logger.error(f"Failed to create template: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create template")


@router.get("/stats")
async def get_alert_stats(
    days: int = Query(7, ge=1, le=30),
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get alert statistics
    
    Args:
        days: Number of days to analyze
        
    Returns:
        Alert statistics
    """
    try:
        # Get recent alerts
        all_alerts = alert_manager.alert_history
        cutoff = datetime.now() - timedelta(days=days)
        recent_alerts = [a for a in all_alerts if a.created_at >= cutoff]
        
        # Calculate stats
        stats = {
            "period_days": days,
            "total_alerts": len(recent_alerts),
            "by_type": {},
            "by_priority": {},
            "by_status": {},
            "by_channel": {},
            "average_per_day": len(recent_alerts) / days if days > 0 else 0
        }
        
        # Count by type
        for alert_type in AlertType:
            count = len([a for a in recent_alerts if a.alert_type == alert_type])
            if count > 0:
                stats["by_type"][alert_type.value] = count
        
        # Count by priority
        for priority in AlertPriority:
            count = len([a for a in recent_alerts if a.priority == priority])
            if count > 0:
                stats["by_priority"][priority.value] = count
        
        # Count by status
        for status in AlertStatus:
            count = len([a for a in recent_alerts if a.status == status])
            if count > 0:
                stats["by_status"][status.value] = count
        
        # Count by channel
        channel_counts = {}
        for alert in recent_alerts:
            for channel in alert.channels:
                channel_counts[channel.value] = channel_counts.get(channel.value, 0) + 1
        stats["by_channel"] = channel_counts
        
        # Success rate
        sent_alerts = [a for a in recent_alerts if a.status in [AlertStatus.SENT, AlertStatus.DELIVERED]]
        stats["success_rate"] = len(sent_alerts) / len(recent_alerts) if recent_alerts else 0
        
        # Acknowledgment rate
        ack_alerts = [a for a in recent_alerts if a.status == AlertStatus.ACKNOWLEDGED]
        stats["acknowledgment_rate"] = len(ack_alerts) / len(recent_alerts) if recent_alerts else 0
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get alert stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to calculate statistics")


@router.post("/start")
async def start_alert_processing(
    current_user=Depends(get_current_user)
) -> Dict[str, str]:
    """
    Start alert processing
    
    Returns:
        Status message
    """
    try:
        await alert_manager.start_processing()
        return {"message": "Alert processing started"}
        
    except Exception as e:
        logger.error(f"Failed to start alert processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start processing")


@router.post("/stop")
async def stop_alert_processing(
    current_user=Depends(get_current_user)
) -> Dict[str, str]:
    """
    Stop alert processing
    
    Returns:
        Status message
    """
    try:
        await alert_manager.stop_processing()
        return {"message": "Alert processing stopped"}
        
    except Exception as e:
        logger.error(f"Failed to stop alert processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to stop processing")


@router.ws("/ws")
async def alerts_websocket(websocket):
    """
    WebSocket endpoint for real-time alerts
    """
    await websocket.accept()
    
    # Add connection to manager
    alert_manager.add_websocket_connection(websocket)
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to alert stream",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive
        while True:
            # Wait for client messages (ping/pong)
            data = await websocket.receive_text()
            
            # Echo back as pong
            if data == "ping":
                await websocket.send_text("pong")
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Remove connection
        alert_manager.remove_websocket_connection(websocket)
        await websocket.close()