
/* LTV 산출 방법
LTV = 이익×거래 기간(라이프타임)×할인율(현재 가치 계수)
LTV = 고객의 연간 거래액 × 수익률 × 고객 지속 연수
LTV = 고객의 평균 구매 단가 × 평균 구매 횟수
LTV = (매출액 - 매출 원가) ÷ 구매자 수
LTV = 평균 구매 단가 × 구매 빈도 × 계속 구매 기간
LTV = (평균 구매 단가 × 구매 빈도 × 계속 구매 기간) -(신규 획득 비용) + 고객 유지 비용)
LTV = ARPU / 이탈률
*/

-- LTV를 알면 신규고객 획득 비용을 파악할 수 있음
-- 상한 CPA(목표 CPA,Cost Per Action) = LTV X 매출 총이익

-- LTV(Life Time Value, 고객 평생 가치)를 구하기 위해 with문으로 사용될 DAU - 신규, 복귀, 이탈 계산하기
with tm1 as
(
select 
	date(payment_date) as,
	customer_id,
	count(payment_id) over(partition by customer_id order by payment_id) as payment_count,
	amount,
	-- 신규유저
	case
		when lag(date(payment_date)) over(partition by customer_id order by date(payment_date)) is null 
		then 1 else 0
	end as new_YN,
	-- 복귀유저
	case 
		when (strftime('%j',date(payment_date)) - strftime('%j',lag(date(payment_date)) over(partition by customer_id order by date(payment_date)))) > 14 
		then 1 else 0
	end as comeback_YN,
	-- 복귀까지 이탈 일
	strftime('%j',date(payment_date)) - strftime('%j',lag(date(payment_date)) over(partition by customer_id order by date(payment_date))) as comeback_Day,
	-- 이탈유저
	case
		when  
			(ifnull(strftime('%j', lead(date(payment_date)) over(partition by customer_id order by date(payment_date))), max(payment_date) over())) > 14
			- strftime('%j',date(payment_date))
		then 1 else 0
	end out_YN
FROM 
	payment
)
-- LTV를 구하기 위한 데이터 산출
SELECT
	date(payment_date),
	count(distinct customer_id) dau,
	sum(cast(amount as bigint)) amount,
	count(case when amount > 0 then customer_id else null end) buy_users,
	(sum(cast(amount as bigint)) * 1.0) / count(distinct customer_id) as arpu,
	(count(case when amount > 0 then customer_id else null end) * 1.0) / count(distinct customer_id) as bur,
	(sum(cast(amount as bigint)) * 1.0) / count(case when amount > 0 then customer_id else null end) as arppu,             
	(count(case when out_YN = 1 then customer_id else null end) * 1.0) / count(distinct customer_id) churn_rate
FROM tm1
WHERE 1=1
and new_YN = 1
group by date(payment_date)
order by 1;
