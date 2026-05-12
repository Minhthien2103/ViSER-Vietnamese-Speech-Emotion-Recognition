import pandas as pd

def thong_ke_gioi_tinh_va_cam_xuc(json_path):
	try:
		df = pd.read_json(json_path)
	except FileNotFoundError:
		print(f"Lỗi: Không tìm thấy file '{json_path}'.")
		return

	print("="*50)
	print("🚻🎭 THỐNG KÊ THEO GIỚI TÍNH & CẢM XÚC")
	print("="*50)
	
	# Gom nhóm theo Giới tính trước, Cảm xúc sau
	stats = df.groupby(['accent','gender', 'emotion']).agg(
		Số_lượng_file=('emotion', 'count'),
		Tổng_thời_lượng_giây=('duration', 'sum')
	)
	
	# Format lại cột thời lượng cho dễ nhìn
	stats['Tổng_thời_lượng_giây'] = stats['Tổng_thời_lượng_giây'].round(2)
	
	print(stats)

if __name__ == "__main__":
	# Thay tên file json thực tế của bạn vào đây
	PATH_TO_JSON = 'data/SENSE_metadata.json' 
	thong_ke_gioi_tinh_va_cam_xuc(PATH_TO_JSON)