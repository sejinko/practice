-- ���� 1��
select * from rental limit 10;

with TMP1 as
(
select 
	customer_id,
	count(rental_id)
from
	rental r 
group by
	customer_id
order by 
	count(rental_id) desc
)
select
	c.customer_id,
	c.first_name,
	c.last_name
from
	tmp1 t
	join customer c on t.customer_id = c.customer_id 
limit 1;


-- ���� 2��
select * from film limit 10;

/*
 x <= 60 : short
 x < 60 <= 120 : middle
 x > 120 : long
*/

select 
	case
		when length <= 60 then 'short'
		when length > 60 and length <=120 then 'middle'
		when length > 120 then 'long'
	end as length_type,
	count(case
		when length <= 60 then 'short'
		when length > 60 and length <=120 then 'middle'
		when length > 120 then 'long'
	end)
from 
	film f
group by
	case
		when length <= 60 then 'short'
		when length > 60 and length <=120 then 'middle'
		when length > 120 then 'long'
	end;
	

-- ���� 3��
select * from film limit 10;


with TM1 as
(
select 
	film_id,
	case 
		when f.rating = 'G' then 'general Audiences'
		when f.rating ='PG' then 'Parental Guidance Suggested.'
		when f.rating ='PG-13' then 'Parents Strongly Cautioned'
		when f.rating ='R' then 'Restricted'
		when f.rating ='NC-17' then 'then no One 17 and under Admitted'
	end as ������,
	case 
		when f.rating ='G' then '��� ���ɴ� ��û����'
		when f.rating ='PG' then '��� ���ɴ� ��û�����ϳ�, �θ��� ������ �ʿ�'
		when f.rating ='PG-13' then '13�� �̸��� �Ƶ����� ������ �� �� ������, �θ��� ���Ǹ� ����'
		when f.rating ='R' then '17�� �Ǵ� ���̻��� ����'
		when f.rating ='NC-17' then '17�� ���� ��û �Ұ�'
	end as �ѱ۸�
from
	film f
)
select
	������ ||' ('|| �ѱ۸� || ')' as full_name
from 
	TM1;


-- ���� 4	��
select * from rental limit 10;

select
	customer_id,
	count(rental_id),
	case 
		when count(rental_id) >= 31 then 'A'
		when count(rental_id) >=21 and count(rental_id) < 31 then 'B'
		when count(rental_id) >= 11 and count(rental_id) < 20 then 'C'
		when count(rental_id) < 10 then 'D'
	end as grade
from 
	rental r 
group by
	customer_id;
	

-- ���� 5��
select * from customer limit 10;

select 
	first_name,
	(case 
		when first_name like 'A@' then 'A'
		when first_name like 'B@' then 'B'
		when first_name like 'C@' then 'C'
		else 'Other'
	end) as flag
from
	customer;
--???


-- ���� 6��
select * from payment limit 10;



	
with TMP1 as
(
select 
	customer_id,
	staff_id,
	to_char(p.payment_date, 'yyyymm') as p_date
from
	payment p
where
	to_char(p.payment_date, 'yyyymm') >= '200701'
and	to_char(p.payment_date, 'yyyymm') <= '200703'
)	
select 
	customer_id,
	(case 
		when staff_id = 2 then 'Y'
		else 'N'
	end) as staff_2
from
	TMP1;
	

-- 7��
select * from payment limit 10;

with TMP1 as
(
select 
	customer_id,
	payment_date,
	to_char(payment_date, 'mm') as mm
from
	payment
)
select
	customer_id, 
	payment_date,
	(case
		when mm >= '01' and mm < '03' then 'Q1'
		when mm >= '04' and mm < '06' then 'Q2'
		when mm >= '07' and mm < '09' then 'Q3'
		when mm >= '10' and mm < '12' then 'Q12'
	end) as Q
from 
	TMP1;
	

-- ���� 8��
select * from rental limit 10;



-- ���� 9��
select * from rental limit 10;

select 
	staff_id,
	return_date,
	count(rental_id) over(partition by staff_id) as flag
	(case 
		when count(rental_id) over(partition by staff_id) >= 0 and count(rental_id) over(partition by staff_id) < 500 then 'under_500'
		when count(rental_id) over(partition by staff_id) >= 500 and count(rental_id) over(partition by staff_id) < 3000 then 'under_3000'
		when count(rental_id) over(partition by staff_id) >= 3000 and count(rental_id) over(partition by staff_id) < 99999 then 'over_3001'
	end) as dd
from
	rental;
	

-- ���� 10��
select * from staff limit 10;


with TMP1 as
(
select 
	staff_id,
	password as old_pw
	
from
	staff s 
)
select
