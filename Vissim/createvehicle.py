'''
生成AV时可能存在位置重叠的背景车，需要把背景车删除
'''
All_Vehicles = Vissim.Net.Vehicles.GetAll()

for i in range(1,min(3,len(All_Vehicles))):
    #最后一辆车获取
    LastVehicle = All_Vehicles[len(All_Vehicles)-i]
    LastID = LastVehicle.AttValue('No')
    LastVehicleLane=int(LastVehicle.AttValue('Lane').split('-')[1])
    LastVehiclePosition=LastVehicle.AttValue('Pos')
    if LastVehicleLane == 2 and  LastVehiclePosition<=12.5:
        Vissim.Net.Vehicles.RemoveVehicle(LastID)
    
vehicle_type =630
desired_speed = 120# unit according to the user setting in Vissim [km/h or mph]
link = 1
lane = 2
xcoordinate = 0  # unit according to the user setting in Vissim [m or ft]
interaction = True# optional boolean
new_Vehicle = Vissim.Net.Vehicles.AddVehicleAtLinkPosition(vehicle_type, link, lane, xcoordinate, desired_speed, interaction)
new_Vehicle.SetAttValue('Speed', desired_speed)