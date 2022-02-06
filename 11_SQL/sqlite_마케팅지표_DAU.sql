-- DAU(Daily Active User)
-- 신규유져
select 
	date(payment_date),
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id),
	amount,

	case
		when lag(date(payment_date)) over(partition by customer_id order by date(payment_date)) is null 
		then 1 else 0
	end as new_YN
from 
	payment;


-- 복귀유저, 14는 이탈일이라 정한 일자임
select 
	date(payment_date),
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id),
	amount,
	case 
		when datediff(date(payment_date), lag(date(payment_date))) over(partition by customer_id order by date(payment_date)) > 14 
		then 1 else 0
	end as comeback_YN
from 
	payment;
	




-- MAU(Mounthly Active User)