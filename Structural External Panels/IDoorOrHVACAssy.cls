VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "IDoorOrHVACAssy"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Public StartComp As IComp
Public EndComp As IComp

Public DoororHVACLength As Double
Public DoororHVACWidth As Double

Public IsDoor As Boolean

Public cChannelCompList As IArrListObject
Public zChannelComplist As IArrListObject
Public lAngleComplist As IArrListObject

Private Sub Class_Initialize()
    
    Set cChannelCompList = New IArrListObject
    Set zChannelComplist = New IArrListObject
    Set lAngleComplist = New IArrListObject
    
End Sub

Function AddToChannelList(oComp As IComp, Optional IsZChannel As Boolean = False, Optional IsLAngle As Boolean = False) As Boolean
    
    AddToChannelList = False

    If Abs(StartComp.xMax - oComp.xMin) <= 0.001 Or Abs(EndComp.xMin - oComp.xMax) <= 0.001 Then
    
        AddToChannelList = True
        
        If IsLAngle Then
  
            lAngleComplist.AddtoList oComp
        
        Else
        
            If IsZChannel Then
                
                zChannelComplist.AddtoList oComp
                
            Else
                cChannelCompList.AddtoList oComp
            
            End If
            
        End If
        
    End If
    
End Function
